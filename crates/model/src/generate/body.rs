//! The body cage: one ring grid from the hips to the crown, with the arm, eye, nose, mouth and ear
//! blocks cut out and refilled, the crotch split into the legs around the perineum and the genital
//! attachment loop (ADR-039), and the limbs attached.
//!
//! The grid has [`COLS`] columns, counter-clockwise around +Y seen from above: column 0 is the front
//! midline, 6 the left side (+X), 12 the back, 18 the right side. Rows are horizontal-ish loops;
//! their front, side and back heights differ (the neck-base loop runs from the suprasternal notch to
//! the seventh cervical vertebra, the jaw loop passes under the chin). Torso and neck rows are
//! superellipse sections; head rows come from an authored table of the face (see [`HEAD`]).

use humanart_math::Vec3;

use super::anatomy::{Anatomy, CROTCH_HEIGHT, TRAGION_TO_VERTEX};
use super::cage::{Cage, Tag};
use super::face;
use super::genitals::{self, GenitalLoops, Variant};
use super::limbs;
use super::regions::{C, R};
use crate::error::{ModelError, Result};

/// Columns of the ring grid.
pub const COLS: usize = 24;
/// Rows of the ring grid.
pub const ROWS: usize = 27;
/// Torso rows T0..T11 are rows 0..=11; the neck rows N0, N1 are 12, 13; head rows H0..H12 are
/// 14..=26.
pub const ROW_N0: usize = 12;
/// The first head row (H0, the jaw loop under the chin).
pub const ROW_H0: usize = 14;

/// A rectangular block of grid faces: face rows `r0..r1`, face columns `c0..c0 + w` (mod COLS).
#[derive(Clone, Copy, Debug)]
pub struct Block {
    /// First vertex row.
    pub r0: usize,
    /// Last vertex row.
    pub r1: usize,
    /// First vertex column.
    pub c0: usize,
    /// Width in faces.
    pub w: usize,
}

impl Block {
    fn col(&self, i: usize) -> usize {
        (self.c0 + i) % COLS
    }

    /// Whether the grid face `(r, c)` (between rows r, r+1 and columns c, c+1) is inside.
    fn has_face(&self, r: usize, c: usize) -> bool {
        r >= self.r0 && r < self.r1 && (c + COLS - self.c0) % COLS < self.w
    }

    /// Whether grid vertex `(r, c)` is strictly inside (not on the boundary).
    fn has_interior(&self, r: usize, c: usize) -> bool {
        let d = (c + COLS - self.c0) % COLS;
        r > self.r0 && r < self.r1 && d > 0 && d < self.w
    }

    /// The boundary as `(row, col)` pairs, counter-clockwise seen from outside.
    pub fn boundary(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0..self.w {
            out.push((self.r0, self.col(i)));
        }
        for r in self.r0..self.r1 {
            out.push((r, self.col(self.w)));
        }
        for i in (1..=self.w).rev() {
            out.push((self.r1, self.col(i)));
        }
        for r in (self.r0 + 1..=self.r1).rev() {
            out.push((r, self.col(0)));
        }
        out
    }
}

/// The blocks cut out of the grid.
pub const ARM_L: Block = Block { r0: 7, r1: 10, c0: 5, w: 2 };
/// Right arm block.
pub const ARM_R: Block = Block { r0: 7, r1: 10, c0: 17, w: 2 };
/// Mouth block (H3..H5, columns -2..2).
pub const MOUTH: Block = Block { r0: 17, r1: 19, c0: 22, w: 4 };
/// Nose block (H5..H7, columns -1..1).
pub const NOSE: Block = Block { r0: 19, r1: 21, c0: 23, w: 2 };
/// Left eye block (H7..H9, columns 1..3).
pub const EYE_L: Block = Block { r0: 21, r1: 23, c0: 1, w: 2 };
/// Right eye block.
pub const EYE_R: Block = Block { r0: 21, r1: 23, c0: 21, w: 2 };
/// Left ear block (H6..H8, columns 5..7).
pub const EAR_L: Block = Block { r0: 20, r1: 22, c0: 5, w: 2 };
/// Right ear block.
pub const EAR_R: Block = Block { r0: 20, r1: 22, c0: 17, w: 2 };

/// Navel block (T1..T3, columns -1..1): the omphalion sits at T2 (ANSUR II `waistheightomphalion`
/// 1018 mm, sex midpoint).
pub const NAVEL: Block = Block { r0: 1, r1: 3, c0: 23, w: 2 };
/// Left nipple block (T5..T7, columns 1..3): the nipple sits near T6, column 2.
pub const NIPPLE_L: Block = Block { r0: 5, r1: 7, c0: 1, w: 2 };
/// Right nipple block.
pub const NIPPLE_R: Block = Block { r0: 5, r1: 7, c0: 21, w: 2 };

const BLOCKS: [Block; 11] = [ARM_L, ARM_R, MOUTH, NOSE, EYE_L, EYE_R, EAR_L, EAR_R, NAVEL, NIPPLE_L, NIPPLE_R];

/// One ring of a [`pit`] fill: horizontal and vertical radius (mm), height along the surface
/// normal (mm, negative goes in), and the region of the faces from the previous ring to it.
type PitRing = (f32, f32, f32, R);

/// The navel (umbilicus): a soft rim, the fold rolling in, and a pit about 11 mm deep, a little
/// taller than wide. Design values (**unsourced**).
const NAVEL_RINGS: [PitRing; 6] = [
    (24.0, 24.0, 0.0, R::Abdomen),
    (8.0, 9.5, 0.6, R::Abdomen),
    (5.5, 7.0, -2.5, R::Navel),
    (4.2, 5.2, -6.0, R::Navel),
    (3.2, 4.0, -9.0, R::Navel),
    (2.0, 2.6, -10.5, R::Navel),
];

/// The areola (15 mm radius, gently domed) and the nipple (4.5 mm radius, 5 mm proud): a
/// sex-neutral size; the identity axes `semantic.nipple-size` / `semantic.areola-size` scale them.
/// Design values (**unsourced**).
fn nipple_rings(side: f32) -> [PitRing; 7] {
    let (areola, nipple) = (R::sided(R::AreolaL, R::AreolaR, side), R::sided(R::NippleL, R::NippleR, side));
    [
        (30.0, 30.0, 0.0, R::Chest),
        (15.0, 15.0, 0.4, R::Chest),
        (9.0, 9.0, 1.1, areola),
        (4.8, 4.8, 1.5, areola),
        (4.4, 4.4, 3.0, nipple),
        (3.6, 3.6, 4.3, nipple),
        (2.2, 2.2, 5.0, nipple),
    ]
}

/// The nipple's height in its block's patch coordinates: 1.233 m, near ANSUR II `chestheight`
/// (1231 mm, sex midpoint).
pub const NIPPLE_T: f32 = 0.66;

/// Lagrange weights of the nodes 0, 1, 2 at `s`.
fn quad_weights(s: f32) -> [f32; 3] {
    [(s - 1.0) * (s - 2.0) * 0.5, -s * (s - 2.0), s * (s - 1.0) * 0.5]
}

/// Fills a 2 x 2 block with concentric rings around a point of the designed surface: each ring
/// lies on the biquadratic patch through the block's nine designed grid points, lifted along the
/// surface normal, and the innermost ring is capped. `centre` is the pit's centre in patch
/// coordinates (0..2, 0..2). Returns the rings, outermost (the block boundary) first.
fn pit(
    cage: &mut Cage,
    a: &Anatomy,
    block: &Block,
    hole: &[u32],
    centre: (f32, f32),
    rings: &[PitRing],
    chart: C,
) -> Result<Vec<Vec<u32>>> {
    if hole.len() != 8 {
        return Err(ModelError::invalid("pit block", hole.len(), "pit block ring must have 8 vertices"));
    }
    let node = |i: usize, j: usize| grid_point(a, block.r0 + j, block.col(i));
    let patch = |s: f32, t: f32| {
        let (ws, wt) = (quad_weights(s), quad_weights(t));
        let mut p = Vec3::ZERO;
        for (j, &w_t) in wt.iter().enumerate() {
            for (i, &w_s) in ws.iter().enumerate() {
                p += node(i, j) * (w_s * w_t);
            }
        }
        p
    };
    let (cs, ct) = centre;
    let c = patch(cs, ct);
    let ds = (patch(cs + 0.05, ct) - patch(cs - 0.05, ct)) / 0.1;
    let dt = (patch(cs, ct + 0.05) - patch(cs, ct - 0.05)) / 0.1;
    // The block's columns run counter-clockwise seen from above: +s is to the left side of the
    // body on the front; U is horizontal, V up, N out of the body.
    let mut n = ds.cross(dt).normalize_or_zero();
    if n.dot(c - Vec3::new(0.0, c.y, -0.02)) < 0.0 {
        n = -n;
    }
    let v = (Vec3::Y - n * n.y).normalize_or_zero();
    let u = v.cross(n);
    // Evenly spaced ring slots, turning the way the boundary does.
    let boundary: Vec<f32> = hole
        .iter()
        .map(|&h| {
            let d = cage.pos[h as usize] - c;
            d.dot(v).atan2(d.dot(u))
        })
        .collect();
    let wrap = |x: f32| (x + 3.0 * std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    let turn: f32 = (0..8).map(|k| wrap(boundary[(k + 1) % 8] - boundary[k])).sum();
    let step = turn.signum() * std::f32::consts::TAU / 8.0;
    // Slots at multiples of 45 degrees (a set the left-right mirror maps onto itself), the first at
    // the one nearest the first boundary vertex.
    let base = (boundary[0] / (std::f32::consts::TAU / 8.0)).round() * std::f32::consts::TAU / 8.0;
    let angles: Vec<f32> = (0..8).map(|k| base + step * k as f32).collect();
    // Patch coordinates of a tangent offset (metres along U and V).
    let (su, tv) = (1.0 / ds.dot(u).abs().max(1e-4), 1.0 / dt.dot(v).abs().max(1e-4));
    let (sign_s, sign_t) = (ds.dot(u).signum(), dt.dot(v).signum());
    // The pit or the nipple is its own UV chart: flattened with the surrounding skin it folds.
    let chart_of = |region: R| match region {
        R::Navel => C::Navel,
        R::NippleL => C::NippleL,
        R::NippleR => C::NippleR,
        _ => chart,
    };
    let mut out = vec![hole.to_vec()];
    for &(rx, ry, h, region) in rings {
        let points: Vec<Vec3> = angles
            .iter()
            .map(|&th| {
                let (x, y) = (th.cos() * rx * 0.001, th.sin() * ry * 0.001);
                let on = patch(cs + sign_s * x * su, ct + sign_t * y * tv);
                on + n * (h * 0.001)
            })
            .collect();
        let ring = cage.add_all(&points);
        let prev = out.last().cloned().unwrap_or_default();
        cage.bridge(&prev, &ring, Tag::new(region.id(), chart_of(region).id()));
        out.push(ring);
    }
    // Close with a fan to the centre, carried on a little past the last ring (a dome or a pit
    // bottom rather than a flat cap).
    let last = out.last().cloned().unwrap_or_default();
    let (region, h_last, h_prev) = match rings {
        [.., p, l] => (l.3, l.2, p.2),
        [l] => (l.3, l.2, 0.0),
        [] => (R::Torso, 0.0, 0.0),
    };
    let apex = cage.add(patch(cs, ct) + n * ((h_last + (h_last - h_prev) * 0.3) * 0.001));
    for k in 0..last.len() {
        cage.face(&[last[k], last[(k + 1) % last.len()], apex], Tag::new(region.id(), chart_of(region).id()));
    }
    Ok(out)
}

/// A torso or neck section: heights at the front, side and back, depths of the front, side and back
/// points, half-width, and the superellipse exponents of the front and back halves. Metres.
#[derive(Clone, Copy, Debug)]
struct Section {
    yf: f32,
    ys: f32,
    yb: f32,
    zf: f32,
    zs: f32,
    zb: f32,
    w: f32,
    pf: f32,
    pb: f32,
}

const fn sec(yf: f32, ys: f32, yb: f32, zf: f32, zs: f32, zb: f32, w: f32, pf: f32, pb: f32) -> Section {
    Section { yf, ys, yb, zf, zs, zb, w, pf, pb }
}

/// Torso rows T0..T11 and neck rows N0, N1. The rows that carry an ANSUR II girth are sized so
/// the subdivided surface meets the sex midpoint (test `anat_02_surface_meets_the_ansur_constants`):
/// T0 at the maximum protrusion of the buttocks (`buttockcircumference` 1020, `buttockdepth` 239,
/// `hipbreadth` 350 mm), T2 at the navel (`waistcircumference` 901, `waistbreadth` 313,
/// `waistdepth` 225 mm), T6 at the nipples (`chestcircumference` 1003, `chestdepth` 251 mm). The
/// girth-to-breadth-plus-depth ratios of the survey (1.7 to 1.9) need box-like sections, hence
/// the superellipse exponents of 3 and more; the rows between are design values.
const TORSO: [Section; 14] = [
    sec(0.866, 0.872, 0.850, 0.074, -0.030, -0.165, 0.172, 3.2, 3.6), // T0 pubis, buttocks
    sec(0.950, 0.958, 0.960, 0.094, -0.022, -0.132, 0.165, 2.8, 3.2), // T1 lower abdomen, ASIS
    sec(1.020, 1.025, 1.025, 0.096, -0.020, -0.125, 0.152, 2.8, 2.9), // T2 navel, iliac crest
    sec(1.080, 1.080, 1.080, 0.094, -0.016, -0.112, 0.146, 2.7, 2.8), // T3 waist
    sec(1.140, 1.140, 1.140, 0.100, -0.012, -0.114, 0.150, 2.8, 3.0), // T4 lower ribs
    sec(1.200, 1.200, 1.200, 0.114, -0.010, -0.126, 0.160, 3.4, 3.4), // T5 under the chest
    sec(1.250, 1.250, 1.250, 0.122, -0.008, -0.129, 0.162, 3.8, 3.8), // T6 chest
    sec(1.300, 1.300, 1.300, 0.118, -0.010, -0.131, 0.155, 3.4, 3.4), // T7 armpit
    sec(1.328, 1.335, 1.335, 0.105, -0.010, -0.124, 0.160, 2.8, 2.9), // T8
    sec(1.348, 1.370, 1.372, 0.092, -0.010, -0.120, 0.170, 2.8, 2.9), // T9
    sec(1.366, 1.412, 1.415, 0.068, -0.012, -0.110, 0.178, 2.6, 2.8), // T10 shoulder top
    sec(1.381, 1.433, 1.436, 0.055, -0.020, -0.092, 0.100, 2.4, 2.6), // T11 trapezius
    sec(1.393, 1.451, 1.457, 0.047, -0.022, -0.079, 0.056, 2.1, 2.2), // N0 neck base
    sec(1.436, 1.468, 1.479, 0.041, -0.022, -0.069, 0.049, 2.1, 2.2), // N1 upper neck
];

/// The arm-socket points of the left side, which replace the section points around the arm block.
const ARM_SOCKET: [((usize, usize), [f32; 3]); 10] = [
    ((7, 5), [0.127, 1.295, 0.045]),
    ((7, 6), [0.120, 1.300, -0.010]),
    ((7, 7), [0.122, 1.295, -0.060]),
    ((8, 7), [0.141, 1.334, -0.076]),
    ((9, 7), [0.165, 1.372, -0.070]),
    ((10, 7), [0.180, 1.405, -0.045]),
    ((10, 6), [0.192, 1.413, -0.008]),
    ((10, 5), [0.182, 1.401, 0.030]),
    ((9, 5), [0.168, 1.368, 0.055]),
    ((8, 5), [0.146, 1.330, 0.066]),
];

/// The head table: per head row H0..H12, the left-side points of columns 0..=6 and the back
/// midline (column 12) as `(y, z)`; millimetres in the head frame (origin between the tragions).
///
/// Face measurements (Farkas, sexes averaged): face height n-gn 117, nose height n-sn 52,
/// sn-sto 21, sto-gn 44, bizygomatic 132, bigonial 100, head breadth 150, head length 193.
#[allow(clippy::type_complexity)]
pub const HEAD: [([[f32; 3]; 7], [f32; 2]); 13] = [
    // H0: under the jaw (submental), behind the angle of the mandible, nape.
    (
        [
            [0., -104., 54.],
            [15., -103., 51.],
            [28., -99., 42.],
            [38., -92., 28.],
            [46., -86., 12.],
            [50., -82., -6.],
            [54.5, -76., -27.],
        ],
        [-60., -70.],
    ),
    // H1: menton and the lower border of the mandible to the gonion.
    (
        [
            [0., -94., 88.],
            [15., -93., 84.],
            [28., -88., 68.],
            [39., -82., 50.],
            [48., -73., 26.],
            [53., -63., 4.],
            [59., -56., -22.],
        ],
        [-42., -82.],
    ),
    // H2: pogonion, jaw.
    (
        [
            [0., -81., 101.],
            [14., -80., 97.],
            [27., -75., 84.],
            [39., -68., 66.],
            [49., -58., 42.],
            [55., -49., 16.],
            [62., -44., -16.],
        ],
        [-24., -91.],
    ),
    // H3: mentolabial sulcus (mouth block bottom).
    (
        [
            [0., -69., 95.],
            [13., -68., 93.],
            [28., -65., 84.],
            [41., -58., 68.],
            [51., -47., 45.],
            [60., -37., 18.],
            [66., -34., -13.],
        ],
        [-8., -97.],
    ),
    // H4: stomion level (mouth block middle).
    (
        [
            [0., -48., 100.],
            [14., -48., 97.],
            [35., -47., 80.],
            [46., -42., 66.],
            [55., -32., 44.],
            [64.5, -24., 18.],
            [70., -26., -11.],
        ],
        [8., -100.],
    ),
    // H5: subnasale (mouth block top, nose block bottom).
    (
        [
            [0., -27., 101.],
            [15., -27., 95.],
            [31., -28., 85.],
            [46.5, -23., 72.],
            [57.5, -17., 50.5],
            [67.5, -13., 19.],
            [72.5, -19., -8.],
        ],
        [24., -101.],
    ),
    // H6: alar crease (nose block middle), cheek, below the ear.
    (
        [
            [0., -12., 115.],
            [19., -12., 92.5],
            [33., -11., 86.],
            [48.5, -7., 75.],
            [60., -5., 54.3],
            [69.2, -8., 18.],
            [74., -12., -6.],
        ],
        [38., -100.],
    ),
    // H7: lower orbit (nose block top, eye block bottom), ear root middle.
    (
        [
            [0., 2., 111.],
            [11., 2., 93.],
            [31., -1., 87.],
            [49.5, 1., 75.8],
            [62., 4., 55.5],
            [70.6, 2., 17.],
            [75., 3., -6.],
        ],
        [50., -98.],
    ),
    // H8: canthal level (eye block middle).
    (
        [
            [0., 13., 102.],
            [10., 13., 95.],
            [31., 12., 85.],
            [51., 14., 75.],
            [63., 16., 55.],
            [71., 14., 16.],
            [75., 18., -6.],
        ],
        [62., -95.],
    ),
    // H9: upper orbital rim (eye block top), nasion.
    (
        [
            [0., 26., 97.],
            [12., 28., 95.],
            [32., 31., 92.],
            [50., 30., 77.],
            [62., 30., 55.],
            [70., 28., 20.],
            [74., 30., -6.],
        ],
        [74., -91.],
    ),
    // H10: brow, glabella.
    (
        [
            [0., 42., 101.],
            [16., 43., 100.],
            [34., 45., 94.],
            [50., 43., 80.],
            [61., 44., 58.],
            [69., 46., 26.],
            [73., 46., -5.],
        ],
        [88., -85.],
    ),
    // H11: forehead.
    (
        [
            [0., 65., 99.],
            [19., 66., 97.],
            [36., 66., 89.],
            [51., 64., 75.],
            [61., 64., 54.],
            [68., 68., 28.],
            [73., 70., -4.],
        ],
        [100., -76.],
    ),
    // H12: hairline.
    (
        [
            [0., 88., 88.],
            [19., 88., 85.],
            [37., 87., 76.],
            [51., 85., 62.],
            [59., 84., 42.],
            [65., 86., 20.],
            [69., 88., -6.],
        ],
        [110., -66.],
    ),
];

/// Named vertex loops and points of the cage that later stages need.
#[derive(Clone, Debug, Default)]
pub struct Loops {
    /// The grid: `grid[row][col]`, `u32::MAX` where a block removed the vertex.
    pub grid: Vec<[u32; COLS]>,
    /// Eye fills, left then right.
    pub eyes: [face::EyeLoops; 2],
    /// The mouth fill.
    pub mouth: face::MouthLoops,
    /// The nose fill.
    pub nose: face::NoseLoops,
    /// Ear fills, left then right.
    pub ears: [face::EarLoops; 2],
    /// Arms (and hands), left then right.
    pub arms: [limbs::ArmLoops; 2],
    /// Legs (and feet), left then right.
    pub legs: [limbs::LegLoops; 2],
    /// The crotch path along the midline: the perineum (the lowest point, ANSUR II
    /// `crotchheight`), the anus, the top of the natal cleft (row T0 at the back).
    pub crotch: [u32; 3],
    /// The genital attachment loop (12 vertices, creases) and the variant's rings inside it.
    pub genitals: GenitalLoops,
    /// Navel rings, outermost (the block boundary) first.
    pub navel: Vec<Vec<u32>>,
    /// Nipple and areola rings (left, right), outermost first.
    pub nipples: [Vec<Vec<u32>>; 2],
    /// Cage edges that are creases.
    pub creases: Vec<(u32, u32)>,
}

/// The body cage with its loops.
#[derive(Clone, Debug)]
pub struct BodyCage {
    /// The cage (positions are the designed surface points before the fit).
    pub cage: Cage,
    /// Loops and points.
    pub loops: Loops,
}

/// A point of the head frame: millimetres to world metres.
///
/// The head is authored with the vertex 133 mm above the tragion; above the brows ([`VAULT_FROM`])
/// heights ease down so the vertex lands at [`TRAGION_TO_VERTEX`] (ANSUR II `tragiontopofhead`).
pub fn head_point(a: &Anatomy, x: f32, y: f32, z: f32) -> Vec3 {
    // A quadratic ease (no slope change at VAULT_FROM, so no crease across the forehead).
    let t = ((y - VAULT_FROM) / (AUTHORED_VERTEX - VAULT_FROM)).max(0.0);
    let y = y - (AUTHORED_VERTEX - TRAGION_TO_VERTEX * 1000.0) * t * t;
    a.head_origin + Vec3::new(x, y, z) * 0.001
}

/// Height of the authored vertex above the tragion, mm.
const AUTHORED_VERTEX: f32 = 133.0;
/// Heights above this (mm above the tragion, above the brows) are scaled into the cranial vault.
const VAULT_FROM: f32 = 40.0;

fn section_point(s: &Section, c: usize) -> Vec3 {
    let (k, sign) = if c <= 12 { (c, 1.0) } else { (COLS - c, -1.0) };
    let t = (k as f32 * 15.0).to_radians();
    let (st, ct) = t.sin_cos();
    let front = k <= 6;
    let p = if front { s.pf } else { s.pb };
    // The superellipse point on the ray at angle `t` of the unit section (polar sampling): the
    // columns stay about evenly spaced along the outline even for box-like sections, where the
    // usual `sin^(2/p)` parameter crowds them at the corners.
    let (sa, ca) = (st.abs(), ct.abs());
    let r = (sa.powf(p) + ca.powf(p)).powf(-1.0 / p);
    // The midline columns exactly on x = 0 (sin 180 degrees is not zero in f32).
    let x = if k % 12 == 0 { 0.0 } else { s.w * (sa * r).min(1.0) };
    let zc = (ca * r).min(1.0);
    let z = if front { s.zs + (s.zf - s.zs) * zc } else { s.zs - (s.zs - s.zb) * zc };
    let blend = ct * ct;
    let y = if front { s.ys + (s.yf - s.ys) * blend } else { s.ys + (s.yb - s.ys) * blend };
    Vec3::new(sign * x, y, z)
}

fn head_row_point(a: &Anatomy, h: usize, c: usize) -> Vec3 {
    let (k, sign) = if c <= 12 { (c, 1.0) } else { (COLS - c, -1.0) };
    let (front, back) = HEAD[h];
    let p = if k <= 6 {
        front[k]
    } else {
        // The back of the head: a superellipse quadrant from column 6 to the back midline.
        let [x6, y6, z6] = front[6];
        let t = ((k - 6) as f32 * 15.0).to_radians();
        let ex = 2.0 / 2.3;
        let x = x6 * t.cos().max(0.0).powf(ex);
        let z = z6 + (back[1] - z6) * t.sin().max(0.0).powf(ex);
        let y = y6 + (back[0] - y6) * (1.0 - t.cos());
        [x, y, z]
    };
    head_point(a, sign * p[0], p[1], p[2])
}

/// The designed position of grid vertex `(r, c)`.
pub fn grid_point(a: &Anatomy, r: usize, c: usize) -> Vec3 {
    if r >= ROW_H0 {
        return head_row_point(a, r - ROW_H0, c);
    }
    for &((rr, cc), p) in &ARM_SOCKET {
        if rr == r && cc == c {
            return Vec3::new(p[0], p[1], p[2]);
        }
        if rr == r && COLS - cc == c {
            return Vec3::new(-p[0], p[1], p[2]);
        }
    }
    section_point(&TORSO[r], c)
}

fn grid_tag(r: usize, c: usize) -> Tag {
    // The face between columns c and c+1, as the lower column index of its left-side mirror image.
    let kf = if c < 12 { c } else { COLS - 1 - c };
    let side = if c < 12 { 1.0 } else { -1.0 };
    if r >= ROW_H0 {
        let h = r - ROW_H0;
        let face = kf <= 4 || (kf == 5 && h < 6);
        if !face {
            return Tag::new(R::Scalp.id(), C::Scalp.id());
        }
        let region = if h >= 9 && kf <= 3 {
            R::Forehead
        } else if h >= 7 {
            R::sided(R::TempleL, R::TempleR, side)
        } else if h <= 2 && kf <= 1 {
            R::Chin
        } else if h <= 2 || kf >= 5 {
            R::sided(R::JawL, R::JawR, side)
        } else {
            R::sided(R::CheekL, R::CheekR, side)
        };
        return Tag::new(region.id(), C::Face.id());
    }
    if r >= ROW_N0 {
        let region = match kf {
            0..=2 => R::NeckFront,
            3..=7 => R::sided(R::NeckSideL, R::NeckSideR, side),
            _ => R::NeckBack,
        };
        return Tag::new(region.id(), C::Neck.id());
    }
    let front = kf < 6;
    let chart = if front { C::TorsoFront } else { C::TorsoBack };
    let region = if r == 0 {
        R::Pelvis
    } else if (3..=8).contains(&kf) && r >= 9 {
        R::sided(R::ShoulderL, R::ShoulderR, side)
    } else if !front {
        R::Back
    } else if r <= 3 {
        R::Abdomen
    } else {
        R::Chest
    };
    Tag::new(region.id(), chart.id())
}

/// The crotch points of the left side (the right side is their mirror image), metres: the
/// front of the genitofemoral fold, the side of the perineum, beside the anus. With the perineum
/// and the anus on the midline they bound the octagon between the legs that holds the perineum,
/// the anus and the natal cleft (behind) and the genital site (in front). Design values
/// (**unsourced**) around ANSUR II `crotchheight`.
const CROTCH_FRONT: [f32; 3] = [0.027, 0.846, 0.054];
const CROTCH_SIDE: [f32; 3] = [0.022, 0.818, 0.004];
const CROTCH_BACK: [f32; 3] = [0.009, 0.834, -0.062];
/// The perineum (central tendon), the lowest midline point between the legs: `(y, z)`.
const PERINEUM_Z: f32 = -0.014;
/// The anus: `(y, z)`.
const ANUS: [f32; 2] = [0.828, -0.050];

/// Builds the body cage with the default genital variant.
pub fn build(a: &Anatomy) -> Result<BodyCage> {
    build_with(a, Variant::default())
}

/// Builds the body cage with the given genital variant (ADR-039).
pub fn build_with(a: &Anatomy, variant: Variant) -> Result<BodyCage> {
    let mut cage = Cage::new();
    let mut loops = Loops::default();
    // Grid vertices (none inside a block).
    for r in 0..ROWS {
        let mut row = [u32::MAX; COLS];
        for (c, slot) in row.iter_mut().enumerate() {
            if !BLOCKS.iter().any(|b| b.has_interior(r, c)) {
                *slot = cage.add(grid_point(a, r, c));
            }
        }
        loops.grid.push(row);
    }
    // Grid faces.
    for r in 0..ROWS - 1 {
        for c in 0..COLS {
            if BLOCKS.iter().any(|b| b.has_face(r, c)) {
                continue;
            }
            let g = &loops.grid;
            let c1 = (c + 1) % COLS;
            let quad = [g[r][c], g[r][c1], g[r + 1][c1], g[r + 1][c]];
            if quad.contains(&u32::MAX) {
                return Err(ModelError::invalid("body grid", r * COLS + c, "grid face uses a removed vertex"));
            }
            cage.face(&quad, grid_tag(r, c));
        }
    }
    let ring_of =
        |loops: &Loops, b: &Block| -> Vec<u32> { b.boundary().iter().map(|&(r, c)| loops.grid[r][c]).collect() };
    // Crown cap over the hairline row.
    let top = loops.grid[ROWS - 1].to_vec();
    let crown = head_point(a, 0.0, 133.0, -12.0);
    cage.cap(&top, 6, 6, 3, Tag::new(R::Scalp.id(), C::Scalp.id()), &|s, t, p| {
        let w = (std::f32::consts::PI * s).sin() * (std::f32::consts::PI * t).sin();
        let lift = w.powf(0.55);
        Vec3::new(p.x, p.y + (crown.y - p.y) * lift, p.z + (crown.z - p.z) * lift * 0.5)
    })?;
    // Face features.
    let mouth_ring = ring_of(&loops, &MOUTH);
    loops.mouth = face::mouth(&mut cage, a, &mouth_ring)?;
    let nose_ring = ring_of(&loops, &NOSE);
    loops.nose = face::nose(&mut cage, a, &nose_ring)?;
    for (i, (block, side)) in [(EYE_L, 1.0_f32), (EYE_R, -1.0)].into_iter().enumerate() {
        let ring = ring_of(&loops, &block);
        loops.eyes[i] = face::eye(&mut cage, a, &ring, side)?;
    }
    for (i, (block, side)) in [(EAR_L, 1.0_f32), (EAR_R, -1.0)].into_iter().enumerate() {
        let ring = ring_of(&loops, &block);
        loops.ears[i] = face::ear(&mut cage, &ring, side)?;
    }
    // Navel and nipples.
    let ring = ring_of(&loops, &NAVEL);
    loops.navel = pit(&mut cage, a, &NAVEL, &ring, (1.0, 0.97), &NAVEL_RINGS, C::TorsoFront)?;
    for (i, (block, side)) in [(NIPPLE_L, 1.0_f32), (NIPPLE_R, -1.0)].into_iter().enumerate() {
        let ring = ring_of(&loops, &block);
        loops.nipples[i] = pit(&mut cage, a, &block, &ring, (1.0, NIPPLE_T), &nipple_rings(side), C::TorsoFront)?;
    }
    // Arms.
    for (i, (block, side)) in [(ARM_L, 1.0_f32), (ARM_R, -1.0)].into_iter().enumerate() {
        let ring = ring_of(&loops, &block);
        loops.arms[i] = limbs::arm(&mut cage, a, &ring, side)?;
    }
    // Crotch and legs. The left leg starts on the left half of row T0 (back to front) and the
    // left crotch points (front to back), counter-clockwise around -Y; the right leg on the mirror.
    let g0 = loops.grid[0];
    let crotch = |p: [f32; 3], side: f32| Vec3::new(side * p[0], p[1], p[2]);
    let [lf, lm, lb] = [CROTCH_FRONT, CROTCH_SIDE, CROTCH_BACK].map(|p| cage.add(crotch(p, 1.0)));
    let [rf, rm, rb] = [CROTCH_FRONT, CROTCH_SIDE, CROTCH_BACK].map(|p| cage.add(crotch(p, -1.0)));
    let mut left: Vec<u32> = (0..=12).rev().map(|c| g0[c]).collect();
    left.extend_from_slice(&[lf, lm, lb]);
    let mut right: Vec<u32> = vec![g0[0]];
    right.extend((12..COLS).rev().map(|c| g0[c]));
    right.extend_from_slice(&[rb, rm, rf]);
    loops.legs[0] = limbs::leg(&mut cage, a, &left, 1.0)?;
    loops.legs[1] = limbs::leg(&mut cage, a, &right, -1.0)?;
    // Between the legs: the perineum and the anus on the midline; behind them the natal cleft
    // (two triangles up to row T0's back point), beside them the perineal skin.
    let pm = cage.add(Vec3::new(0.0, CROTCH_HEIGHT, PERINEUM_Z));
    let an = cage.add(Vec3::new(0.0, ANUS[0], ANUS[1]));
    loops.crotch = [pm, an, g0[12]];
    let leg_tag = |region: R, side: f32| Tag::new(region.id(), C::sided(C::LegL, C::LegR, side).id());
    cage.face(&[g0[12], lb, an], leg_tag(R::Pelvis, 1.0));
    cage.face(&[g0[12], an, rb], leg_tag(R::Pelvis, -1.0));
    cage.face(&[lb, lm, pm, an], leg_tag(R::Perineum, 1.0));
    cage.face(&[an, pm, rm, rb], leg_tag(R::Perineum, -1.0));
    // In front: the genital site, a hexagon bridged by a strip of skin to the attachment loop,
    // whose edges are creases so that the body does not depend on the variant inside it.
    let hexagon = [pm, lm, lf, g0[0], rf, rm];
    let attach = cage.add_all(&genitals::attachment_loop());
    cage.bridge_double(&hexagon, &attach, &|i| {
        let region = if i == 1 || i == 11 { R::Perineum } else { R::Pelvis };
        leg_tag(region, if i < 6 { 1.0 } else { -1.0 })
    });
    for k in 0..attach.len() {
        loops.creases.push((attach[k], attach[(k + 1) % attach.len()]));
    }
    loops.genitals = genitals::build(&mut cage, variant, &attach)?;
    Ok(BodyCage { cage, loops })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_boundaries_are_rings() {
        let b = ARM_L.boundary();
        assert_eq!(b.len(), 10);
        assert_eq!(b[0], (7, 5));
        assert_eq!(MOUTH.boundary().len(), 12);
        assert_eq!(MOUTH.boundary()[0], (17, 22));
        assert!(MOUTH.has_interior(18, 0) && !MOUTH.has_interior(18, 22));
        assert!(NOSE.has_face(19, 0) && NOSE.has_face(20, 23) && !NOSE.has_face(21, 0));
    }

    #[test]
    fn sections_are_symmetric() {
        let s = &TORSO[5];
        for c in 1..12 {
            let (p, q) = (section_point(s, c), section_point(s, COLS - c));
            assert!((p.x + q.x).abs() < 1e-6 && (p.y - q.y).abs() < 1e-6 && (p.z - q.z).abs() < 1e-6);
        }
    }
}
