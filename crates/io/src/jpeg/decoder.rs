//! The decoder proper: the marker walk, frame, table and scan headers, baseline and progressive
//! Huffman entropy decoding (T.81 F.2, G.2), and the IDCT into one sample plane per component.
//! [`super::output`] turns the planes into an image.

use super::exif::{self, Exif};
use super::huffman::{BitReader, DecodeTable};
use super::idct::{IdctTable, idct_block, prescale};
use super::tables::{HuffSpec, ZIGZAG};
use super::{ColorModel, FORMAT, Info, MAX_SCANS};
use crate::bytes::Reader;
use crate::error::{Error, Result};
use crate::image::Limits;

fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

const fn unsupported(what: &'static str) -> Error {
    Error::Unsupported { format: FORMAT, what }
}

/// One image component of the frame, with its decoded samples.
#[derive(Debug)]
pub(super) struct Component {
    /// The component identifier of the frame header.
    pub id: u8,
    /// Horizontal sampling factor, 1 to 4.
    pub h: usize,
    /// Vertical sampling factor, 1 to 4.
    pub v: usize,
    /// Quantisation table selector.
    tq: usize,
    /// Width in samples: `ceil(width * h / hmax)`.
    pub width: usize,
    /// Height in samples: `ceil(height * v / vmax)`.
    pub height: usize,
    /// Blocks per row of the MCU-aligned grid; the plane is `8 * bw` samples wide.
    pub bw: usize,
    /// The dequantisation table, latched at the first scan that codes the component.
    idct: Option<IdctTable>,
    /// Whether some scan has coded the component.
    coded: bool,
    /// The DC predictor of the current scan.
    pred: i32,
    /// The samples, `8 * bw` by `8 * bh`.
    pub plane: Vec<u8>,
    /// Progressive frames only: every block's coefficients (natural order), `64 * bw * bh`.
    coefs: Vec<i16>,
}

impl Component {
    /// The plane's row stride.
    pub const fn stride(&self) -> usize {
        self.bw * 8
    }
}

/// The frame: what SOF says, and the components.
#[derive(Debug)]
pub(super) struct Frame {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// SOF2: progressive DCT.
    pub progressive: bool,
    /// The largest horizontal sampling factor.
    pub hmax: usize,
    /// The largest vertical sampling factor.
    pub vmax: usize,
    /// MCUs per row of an interleaved scan.
    mcux: usize,
    /// MCU rows of an interleaved scan.
    mcuy: usize,
    /// The components, in frame-header order.
    pub components: Vec<Component>,
}

/// The kind of a scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScanKind {
    /// Sequential: all 64 coefficients, decoded and transformed at once.
    Sequential,
    /// Progressive, first DC scan with point transform `al`.
    DcFirst { al: u32 },
    /// Progressive DC refinement: one bit per block.
    DcRefine { al: u32 },
    /// Progressive first AC scan of the band `ss..=se`.
    AcFirst { ss: usize, se: usize, al: u32 },
    /// Progressive AC refinement of the band `ss..=se`.
    AcRefine { ss: usize, se: usize, al: u32 },
}

/// One component of a scan: its index in the frame and its tables.
struct ScanComponent<'t> {
    index: usize,
    dc: Option<&'t DecodeTable>,
    ac: Option<&'t DecodeTable>,
}

/// The whole decoder state between markers.
struct Decoder<'a> {
    data: &'a [u8],
    limits: Limits,
    header_only: bool,
    quant: [Option<[u16; 64]>; 4],
    // Boxed: four of each with their 10-bit lookups would be about 50 KB inline.
    dc: [Option<Box<DecodeTable>>; 4],
    ac: [Option<Box<DecodeTable>>; 4],
    restart_interval: u16,
    frame: Option<Frame>,
    jfif: bool,
    adobe: Option<u8>,
    exif: Option<Exif>,
    icc: bool,
    scans: usize,
    info: Option<Info>,
}

/// Reads the marker at `pos`: `0xFF`, any fill `0xFF` bytes, then the code. Returns the code and the
/// position after it.
fn marker_at(data: &[u8], pos: usize) -> Result<(u8, usize)> {
    match data.get(pos) {
        None => return Err(Error::Truncated { format: FORMAT, offset: pos }),
        Some(0xFF) => {}
        Some(_) => return Err(invalid(pos, "expected a marker")),
    }
    let mut i = pos + 1;
    loop {
        match data.get(i) {
            None => return Err(Error::Truncated { format: FORMAT, offset: i }),
            Some(0xFF) => i += 1,
            Some(0x00) => return Err(invalid(pos, "expected a marker")),
            Some(&code) => return Ok((code, i + 1)),
        }
    }
}

/// Decodes the file up to its first scan (`header_only`) or completely. With `header_only` the
/// frame is `None`.
pub(super) fn decode(data: &[u8], limits: &Limits, header_only: bool) -> Result<(Info, Option<Frame>)> {
    if !data.starts_with(&[0xFF, 0xD8]) {
        return Err(invalid(0, "not a JPEG file (no SOI marker)"));
    }
    let mut d = Decoder {
        data,
        limits: *limits,
        header_only,
        quant: [None; 4],
        dc: [None, None, None, None],
        ac: [None, None, None, None],
        restart_interval: 0,
        frame: None,
        jfif: false,
        adobe: None,
        exif: None,
        icc: false,
        scans: 0,
        info: None,
    };
    let mut pos = 2;
    loop {
        let (code, after) = marker_at(data, pos)?;
        let marker_pos = pos;
        pos = after;
        match code {
            0xD9 => break,
            0xD8 => return Err(invalid(marker_pos, "second SOI marker")),
            // Standalone markers: a stray restart marker after a scan, TEM.
            0xD0..=0xD7 | 0x01 => continue,
            _ => {}
        }
        let len = match data.get(pos..pos + 2) {
            Some(&[a, b]) => usize::from(u16::from_be_bytes([a, b])),
            _ => return Err(Error::Truncated { format: FORMAT, offset: pos }),
        };
        if len < 2 {
            return Err(invalid(pos, "segment length under 2"));
        }
        let end = pos + len;
        if end > data.len() {
            return Err(Error::Truncated { format: FORMAT, offset: data.len() });
        }
        let mut seg = Reader::new(&data[..end], FORMAT);
        seg.seek(pos + 2)?;
        pos = end;
        match code {
            0xC0..=0xC2 => d.frame_header(&mut seg, code == 0xC2)?,
            0xC3 => return Err(unsupported("lossless JPEG")),
            0xC5..=0xC7 | 0xCD..=0xCF => return Err(unsupported("hierarchical JPEG")),
            0xC9..=0xCC => return Err(unsupported("arithmetic coding")),
            0xC4 => d.huffman_tables(&mut seg)?,
            0xDB => d.quant_tables(&mut seg)?,
            0xDD => {
                if seg.remaining() != 2 {
                    return Err(seg.invalid("DRI segment length is not 4"));
                }
                d.restart_interval = seg.u16_be()?;
            }
            0xDA => {
                let scan_start = d.scan(&mut seg)?;
                if header_only {
                    break;
                }
                pos = scan_start;
            }
            0xE0..=0xEF => d.application(code, seg.rest()),
            // DNL (the height is never left to it: SOF with height 0 is refused), JPG, JPGn, COM.
            0xDC | 0xC8 | 0xF0..=0xFE => {}
            _ => return Err(invalid(marker_pos, "unknown marker")),
        }
    }
    let Some(info) = d.info.take() else {
        return Err(invalid(pos, "no scan before the end of the image"));
    };
    if header_only {
        return Ok((info, None));
    }
    let Some(mut frame) = d.frame.take() else {
        return Err(invalid(pos, "no frame header"));
    };
    if frame.components.iter().any(|c| !c.coded) {
        return Err(invalid(pos, "a component is never coded in any scan"));
    }
    if frame.progressive {
        for c in &mut frame.components {
            let Some(table) = c.idct.as_ref() else { continue };
            let stride = c.stride();
            for (b, block) in c.coefs.as_chunks::<64>().0.iter().enumerate() {
                let (bx, by) = (b % c.bw, b / c.bw);
                idct_block(block, table, &mut c.plane[by * 8 * stride + bx * 8..], stride);
            }
        }
    }
    Ok((info, Some(frame)))
}

impl Decoder<'_> {
    /// SOF0, SOF1, SOF2.
    fn frame_header(&mut self, seg: &mut Reader<'_>, progressive: bool) -> Result<()> {
        let at = seg.pos();
        if self.frame.is_some() {
            return Err(invalid(at, "second frame header"));
        }
        match seg.u8()? {
            8 => {}
            12 => return Err(unsupported("12-bit samples")),
            _ => return Err(invalid(at, "sample precision is not 8 bits")),
        }
        let height = usize::from(seg.u16_be()?);
        let width = usize::from(seg.u16_be()?);
        if height == 0 {
            return Err(unsupported("image height defined by a DNL marker"));
        }
        self.limits.check_image(FORMAT, width as u64, height as u64)?;
        let count = usize::from(seg.u8()?);
        match count {
            1 | 3 | 4 => {}
            0 => return Err(invalid(at, "frame has no components")),
            _ => return Err(unsupported("component counts other than 1, 3 and 4")),
        }
        if seg.remaining() != 3 * count {
            return Err(invalid(at, "frame header length does not match its component count"));
        }
        let mut specs = [(0_u8, 0_usize, 0_usize, 0_usize); 4];
        for i in 0..count {
            let pos = seg.pos();
            let id = seg.u8()?;
            let hv = seg.u8()?;
            let tq = seg.u8()?;
            let (h, v) = (usize::from(hv >> 4), usize::from(hv & 15));
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                return Err(invalid(pos, "sampling factor outside 1 to 4"));
            }
            if tq > 3 {
                return Err(invalid(pos, "quantisation table selector over 3"));
            }
            if specs[..i].iter().any(|s| s.0 == id) {
                return Err(invalid(pos, "two components with the same identifier"));
            }
            specs[i] = (id, h, v, usize::from(tq));
        }
        let specs = &specs[..count];
        let hmax = specs.iter().map(|s| s.1).max().unwrap_or(1);
        let vmax = specs.iter().map(|s| s.2).max().unwrap_or(1);
        let mcux = width.div_ceil(8 * hmax);
        let mcuy = height.div_ceil(8 * vmax);
        let blocks: usize = specs.iter().map(|s| mcux * s.1 * mcuy * s.2).sum();
        let per_block = if progressive { 64 * 3 } else { 64 };
        self.limits.check_bytes(FORMAT, "decoded planes", blocks as u64 * per_block)?;
        let alloc = !self.header_only;
        let components = specs
            .iter()
            .map(|&(id, h, v, tq)| {
                let (bw, bh) = (mcux * h, mcuy * v);
                Component {
                    id,
                    h,
                    v,
                    tq,
                    width: (width * h).div_ceil(hmax),
                    height: (height * v).div_ceil(vmax),
                    bw,
                    idct: None,
                    coded: false,
                    pred: 0,
                    plane: if alloc { vec![0; bw * bh * 64] } else { Vec::new() },
                    coefs: if alloc && progressive { vec![0; bw * bh * 64] } else { Vec::new() },
                }
            })
            .collect();
        self.frame = Some(Frame { width, height, progressive, hmax, vmax, mcux, mcuy, components });
        Ok(())
    }

    /// DHT: one or more tables.
    fn huffman_tables(&mut self, seg: &mut Reader<'_>) -> Result<()> {
        while !seg.is_empty() {
            let at = seg.pos();
            let tc_th = seg.u8()?;
            let (class, slot) = (tc_th >> 4, usize::from(tc_th & 15));
            if class > 1 || slot > 3 {
                return Err(invalid(at, "bad Huffman table class or destination"));
            }
            let counts: [u8; 16] = seg.array()?;
            let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
            if total > 256 {
                return Err(invalid(at, "Huffman table has over 256 symbols"));
            }
            let symbols = seg.bytes(total)?.to_vec();
            let table = DecodeTable::new(&HuffSpec { counts, symbols }, at)?;
            if class == 0 {
                self.dc[slot] = Some(Box::new(table));
            } else {
                self.ac[slot] = Some(Box::new(table));
            }
        }
        Ok(())
    }

    /// DQT: one or more tables, stored in natural order.
    fn quant_tables(&mut self, seg: &mut Reader<'_>) -> Result<()> {
        while !seg.is_empty() {
            let at = seg.pos();
            let pq_tq = seg.u8()?;
            let (wide, slot) = (pq_tq >> 4, usize::from(pq_tq & 15));
            if wide > 1 || slot > 3 {
                return Err(invalid(at, "bad quantisation table precision or destination"));
            }
            let mut table = [0_u16; 64];
            for &z in &ZIGZAG {
                let q = if wide == 1 { seg.u16_be()? } else { u16::from(seg.u8()?) };
                if q == 0 {
                    return Err(invalid(at, "quantisation table holds a zero"));
                }
                table[usize::from(z)] = q;
            }
            self.quant[slot] = Some(table);
        }
        Ok(())
    }

    /// APPn: JFIF, EXIF, ICC profile, Adobe.
    fn application(&mut self, code: u8, payload: &[u8]) {
        match code {
            0xE0 if payload.starts_with(b"JFIF\0") => self.jfif = true,
            0xE1 if self.exif.is_none() => self.exif = exif::parse(payload),
            0xE2 if payload.starts_with(b"ICC_PROFILE\0") => self.icc = true,
            0xEE if payload.starts_with(b"Adobe") && payload.len() >= 12 => self.adobe = Some(payload[11]),
            _ => {}
        }
    }

    /// The colour model the components are in.
    fn color_model(&self, frame: &Frame) -> ColorModel {
        match frame.components.len() {
            1 => ColorModel::Grey,
            // Adobe transform 0 means RGB; without Adobe or JFIF, so do the component ids R, G, B.
            3 => match self.adobe {
                Some(0) => ColorModel::Rgb,
                None if !self.jfif && frame.components.iter().map(|c| c.id).eq(*b"RGB") => ColorModel::Rgb,
                _ => ColorModel::YCbCr,
            },
            _ => {
                if self.adobe == Some(2) {
                    ColorModel::Ycck
                } else {
                    ColorModel::Cmyk
                }
            }
        }
    }

    /// SOS: checks the header against the frame, then decodes the scan's entropy-coded data (unless
    /// only the header is wanted). Returns the position of the marker after the data.
    fn scan(&mut self, seg: &mut Reader<'_>) -> Result<usize> {
        let at = seg.pos();
        let Some(frame) = self.frame.as_mut() else {
            return Err(invalid(at, "scan before the frame header"));
        };
        self.scans += 1;
        if self.scans > MAX_SCANS {
            return Err(Error::Limit {
                format: FORMAT,
                what: "scan count",
                limit: MAX_SCANS as u64,
                got: self.scans as u64,
            });
        }
        let count = usize::from(seg.u8()?);
        if count == 0 || count > frame.components.len() {
            return Err(invalid(at, "bad scan component count"));
        }
        if seg.remaining() != 2 * count + 3 {
            return Err(invalid(at, "scan header length does not match its component count"));
        }
        let mut members = [(0_usize, 0_usize, 0_usize); 4];
        for i in 0..count {
            let pos = seg.pos();
            let id = seg.u8()?;
            let tables = seg.u8()?;
            let Some(index) = frame.components.iter().position(|c| c.id == id) else {
                return Err(invalid(pos, "scan names an unknown component"));
            };
            if members[..i].iter().any(|m| m.0 == index) {
                return Err(invalid(pos, "scan names a component twice"));
            }
            let (td, ta) = (usize::from(tables >> 4), usize::from(tables & 15));
            if td > 3 || ta > 3 {
                return Err(invalid(pos, "Huffman table selector over 3"));
            }
            members[i] = (index, td, ta);
        }
        let members = &members[..count];
        let ss = usize::from(seg.u8()?);
        let se = usize::from(seg.u8()?);
        let ahal = seg.u8()?;
        let (ah, al) = (u32::from(ahal >> 4), u32::from(ahal & 15));
        let kind = if frame.progressive {
            if ah > 13 || al > 13 || (ah != 0 && al + 1 != ah) {
                return Err(invalid(at, "bad successive approximation"));
            }
            if ss == 0 {
                if se != 0 {
                    return Err(invalid(at, "progressive scan mixes DC and AC"));
                }
                if ah == 0 { ScanKind::DcFirst { al } } else { ScanKind::DcRefine { al } }
            } else {
                if se < ss || se > 63 {
                    return Err(invalid(at, "bad spectral selection"));
                }
                if count != 1 {
                    return Err(invalid(at, "progressive AC scan with several components"));
                }
                if ah == 0 { ScanKind::AcFirst { ss, se, al } } else { ScanKind::AcRefine { ss, se, al } }
            }
        } else {
            if ss != 0 || se != 63 || ahal != 0 {
                return Err(invalid(at, "sequential scan without the full spectrum"));
            }
            ScanKind::Sequential
        };
        if count > 1 && members.iter().map(|m| frame.components[m.0].h * frame.components[m.0].v).sum::<usize>() > 10 {
            return Err(invalid(at, "MCU of over 10 blocks"));
        }
        let needs_dc = matches!(kind, ScanKind::Sequential | ScanKind::DcFirst { .. });
        let needs_ac = matches!(kind, ScanKind::Sequential | ScanKind::AcFirst { .. } | ScanKind::AcRefine { .. });
        for &(index, td, ta) in members {
            if (needs_dc && self.dc[td].is_none()) || (needs_ac && self.ac[ta].is_none()) {
                return Err(invalid(at, "scan uses an undefined Huffman table"));
            }
            let c = &mut frame.components[index];
            if kind == ScanKind::Sequential && c.coded {
                return Err(invalid(at, "component coded in two sequential scans"));
            }
            if c.idct.is_none() {
                let Some(q) = self.quant[c.tq].as_ref() else {
                    return Err(invalid(at, "component uses an undefined quantisation table"));
                };
                c.idct = Some(prescale(q));
            }
            c.coded = true;
            c.pred = 0;
        }
        if self.info.is_none() {
            let frame = self.frame.as_ref().ok_or_else(|| invalid(at, "scan before the frame header"))?;
            let model = self.color_model(frame);
            let mut sampling = [[0_u8; 2]; 4];
            for (s, c) in sampling.iter_mut().zip(&frame.components) {
                *s = [c.h as u8, c.v as u8];
            }
            self.info = Some(Info {
                width: frame.width as u32,
                height: frame.height as u32,
                components: frame.components.len() as u8,
                progressive: frame.progressive,
                color_model: model,
                sampling,
                restart_interval: self.restart_interval,
                jfif: self.jfif,
                adobe_transform: self.adobe,
                exif: self.exif.unwrap_or_default(),
                has_icc_profile: self.icc,
            });
        }
        let after_header = seg.pos();
        if self.header_only {
            return Ok(after_header);
        }
        let comps: Vec<ScanComponent<'_>> = members
            .iter()
            .map(|&(index, td, ta)| ScanComponent { index, dc: self.dc[td].as_deref(), ac: self.ac[ta].as_deref() })
            .collect();
        let frame = self.frame.as_mut().ok_or_else(|| invalid(at, "scan before the frame header"))?;
        decode_scan(self.data, after_header, frame, &comps, kind, usize::from(self.restart_interval))
    }
}

/// Decodes the entropy-coded data of one scan starting at `start`; returns the position of the
/// marker that follows it.
fn decode_scan(
    data: &[u8],
    start: usize,
    frame: &mut Frame,
    comps: &[ScanComponent<'_>],
    kind: ScanKind,
    restart_interval: usize,
) -> Result<usize> {
    let mut br = BitReader::new(data, start);
    let (mcus_x, mcus_y) = if comps.len() == 1 {
        let c = &frame.components[comps[0].index];
        (c.width.div_ceil(8), c.height.div_ceil(8))
    } else {
        (frame.mcux, frame.mcuy)
    };
    let mut eobrun = 0_u32;
    let mut left = restart_interval;
    let mut restarts = 0_u32;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restart_interval != 0 {
                if left == 0 {
                    br.restart(restarts)?;
                    restarts = restarts.wrapping_add(1);
                    left = restart_interval;
                    eobrun = 0;
                    for sc in comps {
                        frame.components[sc.index].pred = 0;
                    }
                }
                left -= 1;
            }
            if comps.len() == 1 {
                block(&mut br, &mut frame.components[comps[0].index], &comps[0], kind, mx, my, &mut eobrun)?;
            } else {
                for sc in comps {
                    let c = &mut frame.components[sc.index];
                    for by in 0..c.v {
                        for bx in 0..c.h {
                            block(&mut br, c, sc, kind, mx * c.h + bx, my * c.v + by, &mut eobrun)?;
                        }
                    }
                }
            }
        }
        if br.overrun() {
            return Err(Error::Truncated { format: FORMAT, offset: data.len() });
        }
    }
    match br.next_marker() {
        Some((_, at)) => Ok(at),
        None => Err(Error::Truncated { format: FORMAT, offset: data.len() }),
    }
}

/// Decodes one block `(bx, by)` of component `c`.
#[inline]
fn block(
    br: &mut BitReader<'_>,
    c: &mut Component,
    sc: &ScanComponent<'_>,
    kind: ScanKind,
    bx: usize,
    by: usize,
    eobrun: &mut u32,
) -> Result<()> {
    let index = by * c.bw + bx;
    if kind == ScanKind::Sequential {
        let mut coef = [0_i16; 64];
        let (Some(dc), Some(ac)) = (sc.dc, sc.ac) else {
            return Err(br.invalid("scan uses an undefined Huffman table"));
        };
        sequential_block(br, dc, ac, &mut c.pred, &mut coef)?;
        let stride = c.stride();
        if let Some(table) = c.idct.as_ref() {
            idct_block(&coef, table, &mut c.plane[by * 8 * stride + bx * 8..], stride);
        }
        return Ok(());
    }
    let Some(coef) = c.coefs.get_mut(index * 64..index * 64 + 64) else {
        return Err(br.invalid("block outside the frame"));
    };
    match kind {
        ScanKind::Sequential => {}
        ScanKind::DcFirst { al } => {
            let Some(dc) = sc.dc else { return Err(br.invalid("scan uses an undefined Huffman table")) };
            let s = u32::from(br.decode(dc)?);
            if s > 11 {
                return Err(br.invalid("DC difference category over 11"));
            }
            c.pred = c.pred.wrapping_add(br.receive_extend(s));
            coef[0] = c.pred.wrapping_shl(al) as i16;
        }
        ScanKind::DcRefine { al } => {
            if br.bit() {
                coef[0] |= (1 << al) as i16;
            }
        }
        ScanKind::AcFirst { ss, se, al } => {
            let Some(ac) = sc.ac else { return Err(br.invalid("scan uses an undefined Huffman table")) };
            ac_first(br, ac, coef, ss, se, al, eobrun)?;
        }
        ScanKind::AcRefine { ss, se, al } => {
            let Some(ac) = sc.ac else { return Err(br.invalid("scan uses an undefined Huffman table")) };
            ac_refine(br, ac, coef, ss, se, al, eobrun)?;
        }
    }
    Ok(())
}

/// A whole sequential block (T.81 F.2.2): DC difference, then run/size coded AC coefficients.
#[inline]
fn sequential_block(
    br: &mut BitReader<'_>,
    dc: &DecodeTable,
    ac: &DecodeTable,
    pred: &mut i32,
    coef: &mut [i16; 64],
) -> Result<()> {
    let s = u32::from(br.decode(dc)?);
    if s > 11 {
        return Err(br.invalid("DC difference category over 11"));
    }
    *pred = pred.wrapping_add(br.receive_extend(s));
    coef[0] = *pred as i16;
    let mut k = 1;
    while k < 64 {
        if let Some((r, value)) = br.ac_fast(ac) {
            k += r;
            if k > 63 {
                return Err(br.invalid("AC coefficients run past the end of the block"));
            }
            coef[usize::from(ZIGZAG[k])] = value as i16;
            k += 1;
            continue;
        }
        let rs = br.decode(ac)?;
        let (r, s) = (usize::from(rs >> 4), u32::from(rs & 15));
        if s == 0 {
            if r != 15 {
                break;
            }
            k += 16;
            continue;
        }
        k += r;
        if k > 63 {
            return Err(br.invalid("AC coefficients run past the end of the block"));
        }
        coef[usize::from(ZIGZAG[k])] = br.receive_extend(s) as i16;
        k += 1;
    }
    if k > 64 {
        return Err(br.invalid("AC coefficients run past the end of the block"));
    }
    Ok(())
}

/// A progressive first AC scan of one block (T.81 G.1.2.2).
#[inline]
fn ac_first(
    br: &mut BitReader<'_>,
    ac: &DecodeTable,
    coef: &mut [i16],
    ss: usize,
    se: usize,
    al: u32,
    eobrun: &mut u32,
) -> Result<()> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = ss;
    while k <= se {
        if let Some((r, value)) = br.ac_fast(ac) {
            k += r;
            if k > se {
                return Err(br.invalid("AC coefficients run past the spectral band"));
            }
            coef[usize::from(ZIGZAG[k])] = value.wrapping_shl(al) as i16;
            k += 1;
            continue;
        }
        let rs = br.decode(ac)?;
        let (r, s) = (u32::from(rs >> 4), u32::from(rs & 15));
        if s == 0 {
            if r < 15 {
                *eobrun = (1 << r) - 1 + br.bits(r);
                break;
            }
            k += 16;
            continue;
        }
        k += r as usize;
        if k > se {
            return Err(br.invalid("AC coefficients run past the spectral band"));
        }
        coef[usize::from(ZIGZAG[k])] = br.receive_extend(s).wrapping_shl(al) as i16;
        k += 1;
    }
    Ok(())
}

/// A progressive AC refinement scan of one block (T.81 G.1.2.3, in the formulation of libjpeg's
/// `decode_mcu_AC_refine`).
#[inline]
fn ac_refine(
    br: &mut BitReader<'_>,
    ac: &DecodeTable,
    coef: &mut [i16],
    ss: usize,
    se: usize,
    al: u32,
    eobrun: &mut u32,
) -> Result<()> {
    let p1 = (1_i32 << al) as i16;
    let m1 = (-1_i32 << al) as i16;
    let refine = |br: &mut BitReader<'_>, c: &mut i16| {
        if br.bit() && *c & p1 == 0 {
            *c = c.wrapping_add(if *c >= 0 { p1 } else { m1 });
        }
    };
    let mut k = ss;
    if *eobrun == 0 {
        while k <= se {
            let rs = br.decode(ac)?;
            let (mut r, s) = (i32::from(rs >> 4), rs & 15);
            let mut value = 0_i16;
            if s != 0 {
                if s != 1 {
                    return Err(br.invalid("refinement coefficient of size other than 1"));
                }
                value = if br.bit() { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = (1 << r) + br.bits(r as u32);
                break;
            }
            while k <= se {
                let c = &mut coef[usize::from(ZIGZAG[k])];
                if *c != 0 {
                    refine(br, c);
                } else {
                    if r == 0 {
                        break;
                    }
                    r -= 1;
                }
                k += 1;
            }
            if value != 0 {
                if k > se {
                    return Err(br.invalid("AC coefficients run past the spectral band"));
                }
                coef[usize::from(ZIGZAG[k])] = value;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= se {
            let c = &mut coef[usize::from(ZIGZAG[k])];
            if *c != 0 {
                refine(br, c);
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}
