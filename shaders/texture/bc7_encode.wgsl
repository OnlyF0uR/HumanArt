// texture/bc7_encode.wgsl: the BC7 compute encoder [SKN-04]. One invocation per 4 x 4 block.
//
// Input: an rgba8unorm view of one mip level (values as stored: sRGB-encoded colour is encoded as
// is, which is what BC7_UNORM_SRGB expects). Output: one vec4<u32> per block (the 16 bytes,
// little-endian words), at `out_offset + y * blocks.x + x`.
//
// A block of one colour is encoded first and alone: mode 5 with the optimal single-colour table
// (humanart_io::bc::tables::BC7_MODE5_SINGLE), every texel at colour index 1, alpha endpoints equal
// to the alpha. It is exact for every 8-bit colour, which the p-bit modes are not when the
// channels disagree in parity.
//
// Method, per other block: every candidate mode is fitted and the one with the smallest weighted squared
// error (in 0..255 units, channel weights from `params.weights`) is packed. A fit of one subset
// takes the principal axis of its texels (power iteration on the covariance), the extreme
// projections as endpoints, quantises them to the mode's precision choosing the p-bits by endpoint
// error, picks each texel's index by projection checked against the neighbouring palette entries
// (palette computed with the decoder's exact integer interpolation), and then refines the
// endpoints by least squares on the chosen weights. Partitions are chosen by a cheap estimate (the
// same line fit with unquantised endpoints) over all 64 (or 16) partitions.
//
// QUALITY (override, [PERF-R5]):
//   0: mode 6, plus mode 5 for blocks with alpha.
//   1: + modes 1 and 3 (opaque) or 7 (alpha) on the best two-subset partition.
//   2: + the second-best partition, modes 0 and 2 (three subsets), modes 4 and 5 with every
//      rotation and index selection, two refinement passes.
// The partition and anchor tables come from humanart_io::bc::tables::gpu_words (binding 2), the
// single-colour table from tables::single_color_gpu_words at SINGLE_BASE (after the BC6H modes).

override QUALITY: u32 = 1u;

// Word offset of the single-colour table: 192 table words and 14 BC6H modes of 24 words before it
// (SINGLE_COLOR_BASE in humanart_render::compress).
const SINGLE_BASE: u32 = 528u;

struct Params {
    blocks: vec2<u32>,
    size: vec2<u32>,
    out_offset: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    weights: vec4<f32>,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_blocks: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<private> W2: array<u32, 4> = array<u32, 4>(0u, 21u, 43u, 64u);
var<private> W3: array<u32, 8> = array<u32, 8>(0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u);
var<private> W4: array<u32, 16> = array<u32, 16>(0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u);

// The block's texels (0..255), the subset of each texel under the current partition, the error
// weights (rotated with the texels for modes 4 and 5), and the bit writer.
var<private> px: array<vec4<f32>, 16>;
var<private> part: array<u32, 16>;
var<private> wgt: vec4<f32>;
var<private> blk: array<u32, 4>;
var<private> bpos: u32;

// The layout parameters of a mode (humanart_io::bc::bc7::MODES); pkind 0: no p-bits, 1: one per
// endpoint, 2: one per subset.
struct ModeDesc {
    ns: u32,
    pb: u32,
    rb: u32,
    isbb: u32,
    cb: u32,
    ab: u32,
    pkind: u32,
    ib: u32,
    ib2: u32,
}

fn mode_desc(m: u32) -> ModeDesc {
    switch m {
        case 0u: { return ModeDesc(3u, 4u, 0u, 0u, 4u, 0u, 1u, 3u, 0u); }
        case 1u: { return ModeDesc(2u, 6u, 0u, 0u, 6u, 0u, 2u, 3u, 0u); }
        case 2u: { return ModeDesc(3u, 6u, 0u, 0u, 5u, 0u, 0u, 2u, 0u); }
        case 3u: { return ModeDesc(2u, 6u, 0u, 0u, 7u, 0u, 1u, 2u, 0u); }
        case 4u: { return ModeDesc(1u, 0u, 2u, 1u, 5u, 6u, 0u, 2u, 3u); }
        case 5u: { return ModeDesc(1u, 0u, 2u, 0u, 7u, 8u, 0u, 2u, 2u); }
        case 6u: { return ModeDesc(1u, 0u, 0u, 0u, 7u, 7u, 1u, 4u, 0u); }
        default: { return ModeDesc(2u, 6u, 0u, 0u, 5u, 5u, 1u, 2u, 0u); }
    }
}

fn weight(bits: u32, i: u32) -> u32 {
    if (bits == 2u) {
        return W2[min(i, 3u)];
    }
    if (bits == 3u) {
        return W3[min(i, 7u)];
    }
    return W4[min(i, 15u)];
}

fn set_partition(ns: u32, p: u32) {
    for (var t = 0u; t < 16u; t++) {
        var s = 0u;
        if (ns == 2u) {
            s = (tables[p] >> t) & 1u;
        } else if (ns == 3u) {
            s = (tables[64u + p] >> (2u * t)) & 3u;
        }
        part[t] = s;
    }
}

fn anchor(ns: u32, p: u32, s: u32) -> u32 {
    if (s == 0u) {
        return 0u;
    }
    let a = tables[128u + p];
    if (ns == 2u) {
        return a & 0xFFu;
    }
    if (s == 1u) {
        return (a >> 8u) & 0xFFu;
    }
    return (a >> 16u) & 0xFFu;
}

fn is_anchor(ns: u32, p: u32, t: u32) -> bool {
    if (t == 0u) {
        return true;
    }
    if (ns == 2u) {
        return t == anchor(2u, p, 1u);
    }
    if (ns == 3u) {
        return t == anchor(3u, p, 1u) || t == anchor(3u, p, 2u);
    }
    return false;
}

fn expand(v: u32, n: u32) -> u32 {
    let x = v << (8u - n);
    return x | (x >> n);
}

// The 8-bit endpoint a quantised endpoint decodes to; channels with no bits decode to 255.
fn dequant(q: vec4<u32>, p: u32, bits: vec4<u32>, hasp: u32) -> vec4<u32> {
    var out = vec4<u32>(255u);
    for (var c = 0u; c < 4u; c++) {
        if (bits[c] > 0u) {
            let v = select(q[c], (q[c] << 1u) | p, hasp == 1u);
            out[c] = expand(v, bits[c] + hasp);
        }
    }
    return out;
}

// The stored value of one channel nearest to `v` with p-bit `p`.
fn quant_channel(v: f32, bits: u32, p: u32, hasp: u32) -> u32 {
    if (bits == 0u) {
        return 0u;
    }
    let top = (1u << bits) - 1u;
    let n = bits + hasp;
    var guess = v / 255.0 * f32(top);
    if (hasp == 1u) {
        guess = (v / 255.0 * f32((1u << n) - 1u) - f32(p)) * 0.5;
    }
    let g = i32(clamp(round(guess), 0.0, f32(top)));
    var best = u32(g);
    var best_e = 1e30;
    for (var d = -1; d <= 1; d++) {
        let cand = g + d;
        if (cand < 0 || cand > i32(top)) {
            continue;
        }
        let c = u32(cand);
        let r = expand(select(c, (c << 1u) | p, hasp == 1u), n);
        let e = abs(f32(r) - v);
        if (e < best_e) {
            best_e = e;
            best = c;
        }
    }
    return best;
}

fn quant_endpoint(v: vec4<f32>, bits: vec4<u32>, p: u32, hasp: u32) -> vec4<u32> {
    return vec4<u32>(
        quant_channel(v.x, bits.x, p, hasp),
        quant_channel(v.y, bits.y, p, hasp),
        quant_channel(v.z, bits.z, p, hasp),
        quant_channel(v.w, bits.w, p, hasp),
    );
}

fn endpoint_error(v: vec4<f32>, q: vec4<u32>, p: u32, bits: vec4<u32>, hasp: u32, mask: vec4<f32>) -> f32 {
    let d = (vec4<f32>(dequant(q, p, bits, hasp)) - v) * mask;
    return dot(d * d, wgt);
}

struct Quant {
    q0: vec4<u32>,
    q1: vec4<u32>,
    p0: u32,
    p1: u32,
}

fn quantize_pair(e0: vec4<f32>, e1: vec4<f32>, bits: vec4<u32>, pkind: u32, mask: vec4<f32>) -> Quant {
    if (pkind == 0u) {
        return Quant(quant_endpoint(e0, bits, 0u, 0u), quant_endpoint(e1, bits, 0u, 0u), 0u, 0u);
    }
    var best = Quant(vec4<u32>(0u), vec4<u32>(0u), 0u, 0u);
    var best_e = 1e30;
    for (var pp = 0u; pp < 4u; pp++) {
        let p0 = pp & 1u;
        let p1 = pp >> 1u;
        if (pkind == 2u && p0 != p1) {
            continue;
        }
        let q0 = quant_endpoint(e0, bits, p0, 1u);
        let q1 = quant_endpoint(e1, bits, p1, 1u);
        let e = endpoint_error(e0, q0, p0, bits, 1u, mask) + endpoint_error(e1, q1, p1, bits, 1u, mask);
        if (e < best_e) {
            best_e = e;
            best = Quant(q0, q1, p0, p1);
        }
    }
    return best;
}

struct Fit {
    err: f32,
    q: Quant,
    idx: array<u32, 16>,
}

// Picks the index of every texel of subset `s` for the quantised endpoints and sums the error.
fn evaluate(s: u32, q: Quant, bits: vec4<u32>, hasp: u32, ib: u32, mask: vec4<f32>) -> Fit {
    var fit: Fit;
    fit.q = q;
    let ai = dequant(q.q0, q.p0, bits, hasp);
    let bi = dequant(q.q1, q.p1, bits, hasp);
    let a = vec4<f32>(ai);
    let d = (vec4<f32>(bi) - a) * mask;
    let dd = dot(d * d, wgt);
    let top = (1u << ib) - 1u;
    var err = 0.0;
    for (var t = 0u; t < 16u; t++) {
        fit.idx[t] = 0u;
        if (part[t] != s) {
            continue;
        }
        var k = 0;
        if (dd > 0.0) {
            let proj = dot((px[t] - a) * mask * wgt, d) / dd;
            k = i32(clamp(round(proj * f32(top)), 0.0, f32(top)));
        }
        var best_k = 0u;
        var best_e = 1e30;
        for (var o = -1; o <= 1; o++) {
            let cand = k + o;
            if (cand < 0 || cand > i32(top)) {
                continue;
            }
            let w = weight(ib, u32(cand));
            let pal = (ai * (64u - w) + bi * w + vec4<u32>(32u)) >> vec4<u32>(6u);
            let e4 = (vec4<f32>(pal) - px[t]) * mask;
            let e = dot(e4 * e4, wgt);
            if (e < best_e) {
                best_e = e;
                best_k = u32(cand);
            }
        }
        fit.idx[t] = best_k;
        err += best_e;
    }
    fit.err = err;
    return fit;
}

struct Line {
    mean: vec4<f32>,
    axis: vec4<f32>,
}

// The mean and the principal axis of the masked texels of subset `s`.
fn principal(s: u32, mask: vec4<f32>) -> Line {
    var mean = vec4<f32>(0.0);
    var n = 0.0;
    for (var t = 0u; t < 16u; t++) {
        if (part[t] == s) {
            mean += px[t];
            n += 1.0;
        }
    }
    mean = mean / max(n, 1.0) * mask;
    var cov = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var t = 0u; t < 16u; t++) {
        if (part[t] == s) {
            let d = px[t] * mask - mean;
            cov += mat4x4<f32>(d * d.x, d * d.y, d * d.z, d * d.w);
        }
    }
    var v = cov[0];
    var largest = cov[0][0];
    for (var c = 1u; c < 4u; c++) {
        if (cov[c][c] > largest) {
            largest = cov[c][c];
            v = cov[c];
        }
    }
    for (var i = 0u; i < 4u; i++) {
        let l = length(v);
        if (l < 1e-9) {
            break;
        }
        v = cov * (v / l);
    }
    let l = length(v);
    return Line(mean, select(vec4<f32>(0.0), v / l, l > 1e-9));
}

struct Pair {
    e0: vec4<f32>,
    e1: vec4<f32>,
    ok: bool,
}

// The least-squares endpoints for the weights the indices of `fit` select.
fn refine(s: u32, fit: Fit, ib: u32, mask: vec4<f32>) -> Pair {
    var aa = 0.0;
    var ab = 0.0;
    var bb = 0.0;
    var d0 = vec4<f32>(0.0);
    var d1 = vec4<f32>(0.0);
    for (var t = 0u; t < 16u; t++) {
        if (part[t] != s) {
            continue;
        }
        let w = f32(weight(ib, fit.idx[t])) / 64.0;
        let iw = 1.0 - w;
        aa += iw * iw;
        ab += iw * w;
        bb += w * w;
        d0 += iw * px[t];
        d1 += w * px[t];
    }
    let det = aa * bb - ab * ab;
    if (abs(det) < 1e-6) {
        return Pair(vec4<f32>(0.0), vec4<f32>(0.0), false);
    }
    let e0 = clamp((bb * d0 - ab * d1) / det * mask, vec4<f32>(0.0), vec4<f32>(255.0));
    let e1 = clamp((aa * d1 - ab * d0) / det * mask, vec4<f32>(0.0), vec4<f32>(255.0));
    return Pair(e0, e1, true);
}

// Fits subset `s` with `bits` per endpoint channel, p-bit kind `pkind` and `ib`-bit indices.
fn fit_subset(s: u32, mask: vec4<f32>, bits: vec4<u32>, pkind: u32, ib: u32, iters: u32) -> Fit {
    let line = principal(s, mask);
    var tmin = 1e30;
    var tmax = -1e30;
    for (var t = 0u; t < 16u; t++) {
        if (part[t] == s) {
            let proj = dot(px[t] * mask - line.mean, line.axis);
            tmin = min(tmin, proj);
            tmax = max(tmax, proj);
        }
    }
    let e0 = clamp(line.mean + line.axis * tmin, vec4<f32>(0.0), vec4<f32>(255.0));
    let e1 = clamp(line.mean + line.axis * tmax, vec4<f32>(0.0), vec4<f32>(255.0));
    let hasp = select(0u, 1u, pkind > 0u);
    var best = evaluate(s, quantize_pair(e0, e1, bits, pkind, mask), bits, hasp, ib, mask);
    for (var i = 0u; i < iters; i++) {
        let r = refine(s, best, ib, mask);
        if (!r.ok) {
            break;
        }
        let f = evaluate(s, quantize_pair(r.e0, r.e1, bits, pkind, mask), bits, hasp, ib, mask);
        if (f.err < best.err) {
            best = f;
        } else {
            break;
        }
    }
    return best;
}

// The error of a partition with unquantised line endpoints and `levels` evenly spaced indices.
fn estimate(ns: u32, p: u32, mask: vec4<f32>, levels: f32) -> f32 {
    set_partition(ns, p);
    var err = 0.0;
    for (var s = 0u; s < ns; s++) {
        let line = principal(s, mask);
        var tmin = 1e30;
        var tmax = -1e30;
        for (var t = 0u; t < 16u; t++) {
            if (part[t] == s) {
                let proj = dot(px[t] * mask - line.mean, line.axis);
                tmin = min(tmin, proj);
                tmax = max(tmax, proj);
            }
        }
        let range = tmax - tmin;
        for (var t = 0u; t < 16u; t++) {
            if (part[t] == s) {
                let proj = dot(px[t] * mask - line.mean, line.axis);
                var q = tmin;
                if (range > 0.0) {
                    q = tmin + round((proj - tmin) / range * (levels - 1.0)) / (levels - 1.0) * range;
                }
                let d = px[t] * mask - (line.mean + line.axis * q);
                err += dot(d * d, wgt);
            }
        }
    }
    return err;
}

// The two partitions (of the first `count`) with the smallest estimate.
fn best_partitions(ns: u32, count: u32, mask: vec4<f32>, levels: f32) -> vec2<u32> {
    var best = vec2<u32>(0u, 0u);
    var e1 = 1e30;
    var e2 = 1e30;
    for (var p = 0u; p < count; p++) {
        let e = estimate(ns, p, mask, levels);
        if (e < e1) {
            e2 = e1;
            best.y = best.x;
            e1 = e;
            best.x = p;
        } else if (e < e2) {
            e2 = e;
            best.y = p;
        }
    }
    return best;
}

// A complete candidate encoding. For modes 4 and 5, q[0] and q[1] hold the colour endpoints in xyz
// and the alpha endpoints in w, idx the colour indices and idx2 the alpha indices.
struct Enc {
    err: f32,
    mode: u32,
    part: u32,
    rot: u32,
    isb: u32,
    q: array<vec4<u32>, 6>,
    p: array<u32, 6>,
    idx: array<u32, 16>,
    idx2: array<u32, 16>,
}

// Modes 0, 1, 2, 3, 6 and 7 on partition `p`.
fn try_partitioned(m: u32, p: u32, iters: u32) -> Enc {
    let md = mode_desc(m);
    set_partition(md.ns, p);
    let bits = vec4<u32>(md.cb, md.cb, md.cb, md.ab);
    let mask = select(vec4<f32>(1.0, 1.0, 1.0, 0.0), vec4<f32>(1.0), md.ab > 0u);
    var enc: Enc;
    enc.err = 0.0;
    enc.mode = m;
    enc.part = p;
    enc.rot = 0u;
    enc.isb = 0u;
    for (var s = 0u; s < md.ns; s++) {
        let f = fit_subset(s, mask, bits, md.pkind, md.ib, iters);
        enc.err += f.err;
        enc.q[2u * s] = f.q.q0;
        enc.q[2u * s + 1u] = f.q.q1;
        enc.p[2u * s] = f.q.p0;
        enc.p[2u * s + 1u] = f.q.p1;
        for (var t = 0u; t < 16u; t++) {
            if (part[t] == s) {
                enc.idx[t] = f.idx[t];
            }
        }
    }
    return enc;
}

// Swaps alpha with channel rot - 1 of the texels and the weights (its own inverse).
fn rotate(rot: u32) {
    if (rot == 0u) {
        return;
    }
    let c = rot - 1u;
    for (var t = 0u; t < 16u; t++) {
        let v = px[t][c];
        px[t][c] = px[t].w;
        px[t].w = v;
    }
    let w = wgt[c];
    wgt[c] = wgt.w;
    wgt.w = w;
}

// Modes 4 and 5: colour and alpha fitted separately, after channel rotation `rot`.
fn try_separate_alpha(m: u32, rot: u32, isb: u32, iters: u32) -> Enc {
    let md = mode_desc(m);
    rotate(rot);
    set_partition(1u, 0u);
    let cib = select(md.ib, md.ib2, isb == 1u);
    let aib = select(md.ib2, md.ib, isb == 1u);
    let fc = fit_subset(0u, vec4<f32>(1.0, 1.0, 1.0, 0.0), vec4<u32>(md.cb, md.cb, md.cb, 0u), 0u, cib, iters);
    let fa = fit_subset(0u, vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<u32>(0u, 0u, 0u, md.ab), 0u, aib, iters);
    rotate(rot);
    var enc: Enc;
    enc.err = fc.err + fa.err;
    enc.mode = m;
    enc.part = 0u;
    enc.rot = rot;
    enc.isb = isb;
    enc.q[0] = vec4<u32>(fc.q.q0.xyz, fa.q.q0.w);
    enc.q[1] = vec4<u32>(fc.q.q1.xyz, fa.q.q1.w);
    enc.idx = fc.idx;
    enc.idx2 = fa.idx;
    return enc;
}

// Appends the low `n` bits of `v`.
fn put(v: u32, n: u32) {
    if (n == 0u) {
        return;
    }
    let word = bpos >> 5u;
    let bit = bpos & 31u;
    let val = v & select((1u << n) - 1u, 0xFFFFFFFFu, n >= 32u);
    if (word < 4u) {
        blk[word] |= val << bit;
        if (bit + n > 32u && word < 3u) {
            blk[word + 1u] |= val >> (32u - bit);
        }
    }
    bpos += n;
}

// Applies the anchor rule (the anchor's index has its top bit clear, by swapping the subset's
// endpoints and inverting its indices) and writes the 128 bits.
fn pack(enc_in: Enc) -> vec4<u32> {
    var e = enc_in;
    let md = mode_desc(e.mode);
    set_partition(md.ns, e.part);
    if (md.ib2 == 0u) {
        let top = (1u << md.ib) - 1u;
        for (var s = 0u; s < md.ns; s++) {
            let a = anchor(md.ns, e.part, s);
            if (((e.idx[a] >> (md.ib - 1u)) & 1u) == 1u) {
                let q = e.q[2u * s];
                e.q[2u * s] = e.q[2u * s + 1u];
                e.q[2u * s + 1u] = q;
                let p = e.p[2u * s];
                e.p[2u * s] = e.p[2u * s + 1u];
                e.p[2u * s + 1u] = p;
                for (var t = 0u; t < 16u; t++) {
                    if (part[t] == s) {
                        e.idx[t] = top - e.idx[t];
                    }
                }
            }
        }
    } else {
        let cib = select(md.ib, md.ib2, e.isb == 1u);
        let aib = select(md.ib2, md.ib, e.isb == 1u);
        if (((e.idx[0] >> (cib - 1u)) & 1u) == 1u) {
            let q0 = e.q[0];
            e.q[0] = vec4<u32>(e.q[1].xyz, q0.w);
            e.q[1] = vec4<u32>(q0.xyz, e.q[1].w);
            for (var t = 0u; t < 16u; t++) {
                e.idx[t] = (1u << cib) - 1u - e.idx[t];
            }
        }
        if (((e.idx2[0] >> (aib - 1u)) & 1u) == 1u) {
            let a0 = e.q[0].w;
            e.q[0].w = e.q[1].w;
            e.q[1].w = a0;
            for (var t = 0u; t < 16u; t++) {
                e.idx2[t] = (1u << aib) - 1u - e.idx2[t];
            }
        }
    }
    blk = array<u32, 4>(0u, 0u, 0u, 0u);
    bpos = 0u;
    put(1u << e.mode, e.mode + 1u);
    put(e.part, md.pb);
    put(e.rot, md.rb);
    put(e.isb, md.isbb);
    let nch = select(3u, 4u, md.ab > 0u);
    for (var c = 0u; c < nch; c++) {
        let n = select(md.cb, md.ab, c == 3u);
        for (var s = 0u; s < md.ns; s++) {
            put(e.q[2u * s][c], n);
            put(e.q[2u * s + 1u][c], n);
        }
    }
    if (md.pkind == 1u) {
        for (var i = 0u; i < 2u * md.ns; i++) {
            put(e.p[i], 1u);
        }
    } else if (md.pkind == 2u) {
        for (var s = 0u; s < md.ns; s++) {
            put(e.p[2u * s], 1u);
        }
    }
    if (md.ib2 == 0u) {
        for (var t = 0u; t < 16u; t++) {
            put(e.idx[t], md.ib - select(0u, 1u, is_anchor(md.ns, e.part, t)));
        }
    } else {
        // Primary (md.ib bits) then secondary (md.ib2 bits); mode 4 with isb = 1 stores the alpha
        // indices as the primary set.
        let swap = e.isb == 1u;
        for (var t = 0u; t < 16u; t++) {
            put(select(e.idx[t], e.idx2[t], swap), md.ib - select(0u, 1u, t == 0u));
        }
        for (var t = 0u; t < 16u; t++) {
            put(select(e.idx2[t], e.idx[t], swap), md.ib2 - select(0u, 1u, t == 0u));
        }
    }
    return vec4<u32>(blk[0], blk[1], blk[2], blk[3]);
}

// The mode 5 encoding of a block whose 16 texels all equal `c` (0..255): exact.
fn single_color(c: vec4<f32>) -> Enc {
    var enc: Enc;
    enc.err = 0.0;
    enc.mode = 5u;
    enc.part = 0u;
    enc.rot = 0u;
    enc.isb = 0u;
    let v = vec4<u32>(c);
    var q0 = vec4<u32>(0u, 0u, 0u, v.w);
    var q1 = vec4<u32>(0u, 0u, 0u, v.w);
    for (var ch = 0u; ch < 3u; ch++) {
        let word = tables[SINGLE_BASE + (v[ch] >> 1u)];
        let entry = (word >> (16u * (v[ch] & 1u))) & 0xFFFFu;
        q0[ch] = entry & 0x7Fu;
        q1[ch] = (entry >> 8u) & 0x7Fu;
    }
    enc.q[0] = q0;
    enc.q[1] = q1;
    for (var t = 0u; t < 16u; t++) {
        enc.idx[t] = 1u;
        enc.idx2[t] = 0u;
    }
    return enc;
}

@compute @workgroup_size(8, 8)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.blocks.x || id.y >= params.blocks.y) {
        return;
    }
    wgt = params.weights;
    var opaque = true;
    for (var t = 0u; t < 16u; t++) {
        let x = min(id.x * 4u + (t & 3u), params.size.x - 1u);
        let y = min(id.y * 4u + (t >> 2u), params.size.y - 1u);
        let c = round(clamp(textureLoad(src, vec2<i32>(i32(x), i32(y)), 0), vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0);
        px[t] = c;
        if (c.w < 255.0) {
            opaque = false;
        }
    }
    var uniform_block = true;
    for (var t = 1u; t < 16u; t++) {
        if (any(px[t] != px[0])) {
            uniform_block = false;
        }
    }
    if (uniform_block) {
        out_blocks[params.out_offset + id.y * params.blocks.x + id.x] = pack(single_color(px[0]));
        return;
    }
    let iters = select(1u, 2u, QUALITY >= 2u);
    var best = try_partitioned(6u, 0u, iters);
    if (best.err > 0.0 && !opaque) {
        let c = try_separate_alpha(5u, 0u, 0u, iters);
        if (c.err < best.err) {
            best = c;
        }
    }
    if (QUALITY >= 1u && best.err > 0.0) {
        if (opaque) {
            let rgb = vec4<f32>(1.0, 1.0, 1.0, 0.0);
            let p2 = best_partitions(2u, 64u, rgb, 8.0);
            var tries = 1u;
            if (QUALITY >= 2u) {
                tries = 2u;
            }
            for (var k = 0u; k < tries; k++) {
                let p = select(p2.x, p2.y, k == 1u);
                let c1 = try_partitioned(1u, p, iters);
                if (c1.err < best.err) {
                    best = c1;
                }
                let c3 = try_partitioned(3u, p, iters);
                if (c3.err < best.err) {
                    best = c3;
                }
            }
            if (QUALITY >= 2u) {
                let p3 = best_partitions(3u, 64u, rgb, 4.0);
                let c2 = try_partitioned(2u, p3.x, iters);
                if (c2.err < best.err) {
                    best = c2;
                }
                let p0 = best_partitions(3u, 16u, rgb, 8.0);
                let c0 = try_partitioned(0u, p0.x, iters);
                if (c0.err < best.err) {
                    best = c0;
                }
            }
        } else {
            let p2 = best_partitions(2u, 64u, vec4<f32>(1.0), 4.0);
            let c7 = try_partitioned(7u, p2.x, iters);
            if (c7.err < best.err) {
                best = c7;
            }
            if (QUALITY >= 2u) {
                let c7b = try_partitioned(7u, p2.y, iters);
                if (c7b.err < best.err) {
                    best = c7b;
                }
            }
        }
    }
    if (QUALITY >= 2u && best.err > 0.0) {
        for (var rot = 0u; rot < 4u; rot++) {
            let c5 = try_separate_alpha(5u, rot, 0u, iters);
            if (c5.err < best.err) {
                best = c5;
            }
            for (var isb = 0u; isb < 2u; isb++) {
                let c4 = try_separate_alpha(4u, rot, isb, iters);
                if (c4.err < best.err) {
                    best = c4;
                }
            }
        }
    }
    out_blocks[params.out_offset + id.y * params.blocks.x + id.x] = pack(best);
}
