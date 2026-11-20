// texture/bc6h_encode.wgsl: the BC6H compute encoder [SKN-04]. One invocation per 4 x 4 block.
//
// Input: a float view (rgba16float or rgba32float) of one mip level; RGB is encoded, alpha ignored.
// Output: one vec4<u32> per block at `out_offset + y * blocks.x + x`.
//
// Domain: texels are converted to half floats and handled as the integers of their bit patterns
// (sign-magnitude for SIGNED). BC6H interpolates linearly in exactly this domain (the decoder's
// "finish" step scales the interpolated value by 31/64, or 31/32 signed, into half bits), so a
// squared error there is a relative (roughly logarithmic) error in the colour, which is the
// standard BC6H error measure.
//
// Method, per region: the principal axis of the region's texels gives the endpoints, they are
// quantised to the mode's endpoint precision (the inverse of the decoder's unquantisation, checked
// against the neighbours), the indices are chosen against the decoder's exact palette, and the
// endpoints are refined by least squares on the chosen weights. Endpoints are then swapped where the
// anchor rule requires it, and transformed modes keep the candidate only if every delta fits its
// bit width. One-region modes 10 to 13 are always tried; two-region modes 0 to 9 on the best
// partitions of 32 (QUALITY >= 1). Mode tables come from binding 2: words 0..192 the BC7 partition
// tables (humanart_io::bc::tables::gpu_words), then 24 words per mode
// (humanart_io::bc::bc6h::gpu_words).
//
// Overrides [PERF-R5]: SIGNED selects BC6H_SF16; QUALITY 0: one-region modes; 1: + two-region modes
// on the best partition; 2: + the second-best partition and two refinement passes.

override SIGNED: bool = false;
override QUALITY: u32 = 1u;

struct Params {
    blocks: vec2<u32>,
    size: vec2<u32>,
    out_offset: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_blocks: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const MODE_BASE: u32 = 192u;
const MODE_WORDS: u32 = 24u;

var<private> W3: array<i32, 8> = array<i32, 8>(0, 9, 18, 27, 37, 46, 55, 64);
var<private> W4: array<i32, 16> = array<i32, 16>(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);

var<private> px: array<vec3<f32>, 16>;
var<private> part: array<u32, 16>;
var<private> blk: array<u32, 4>;
var<private> bpos: u32;

fn weight(ib: u32, i: u32) -> i32 {
    if (ib == 3u) {
        return W3[min(i, 7u)];
    }
    return W4[min(i, 15u)];
}

// A float as the integer of its half-float bit pattern (sign-magnitude when SIGNED).
fn half_int(v: f32) -> f32 {
    var x = v;
    if (!(x == x)) {
        x = 0.0;
    }
    if (SIGNED) {
        x = clamp(x, -65504.0, 65504.0);
    } else {
        x = clamp(x, 0.0, 65504.0);
    }
    let bits = pack2x16float(vec2<f32>(abs(x), 0.0)) & 0x7FFFu;
    return select(f32(bits), -f32(bits), x < 0.0);
}

fn set_partition(regions: u32, p: u32) {
    for (var t = 0u; t < 16u; t++) {
        part[t] = select(0u, (tables[p] >> t) & 1u, regions == 2u);
    }
}

fn is_anchor(regions: u32, p: u32, t: u32) -> bool {
    return t == 0u || (regions == 2u && t == (tables[128u + p] & 0xFFu));
}

// The decoder's unquantisation of one endpoint component of `bits` bits.
fn unquantize(c: i32, bits: u32) -> i32 {
    if (SIGNED) {
        if (bits >= 16u) {
            return c;
        }
        let mag = abs(c);
        var u = 0;
        if (mag == 0) {
            u = 0;
        } else if (mag >= (1 << (bits - 1u)) - 1) {
            u = 0x7FFF;
        } else {
            u = ((mag << 15u) + 0x4000) >> (bits - 1u);
        }
        return select(u, -u, c < 0);
    }
    if (bits >= 15u) {
        return c;
    }
    if (c == 0) {
        return 0;
    }
    if (c == (1 << bits) - 1) {
        return 0xFFFF;
    }
    return ((c << 16u) + 0x8000) >> bits;
}

// The decoder's final scale to half bits, as a signed integer.
fn finish(v: i32) -> i32 {
    if (SIGNED) {
        return select((v * 31) >> 5u, -(((-v) * 31) >> 5u), v < 0);
    }
    return (v * 31) >> 6u;
}

// The endpoint component of `bits` bits that unquantises closest to `u` (interpolation domain).
fn quantize(u: f32, bits: u32) -> i32 {
    var lo = 0;
    var hi = (1 << bits) - 1;
    var guess = 0.0;
    if (SIGNED) {
        hi = select((1 << (bits - 1u)) - 1, 32767, bits >= 16u);
        lo = -hi;
        guess = select(sign(u) * (abs(u) * f32(1u << (bits - 1u)) / 32768.0 - 0.5), u, bits >= 16u);
    } else {
        guess = select(u * f32(1u << bits) / 65536.0 - 0.5, u, bits >= 15u);
    }
    let g = i32(clamp(round(guess), f32(lo), f32(hi)));
    var best = g;
    var best_e = 1e30;
    for (var d = -1; d <= 1; d++) {
        let c = clamp(g + d, lo, hi);
        let e = abs(f32(unquantize(c, bits)) - u);
        if (e < best_e) {
            best_e = e;
            best = c;
        }
    }
    return best;
}

// Half-int domain to interpolation domain.
fn to_interp(h: vec3<f32>) -> vec3<f32> {
    return h * select(64.0 / 31.0, 32.0 / 31.0, SIGNED);
}

struct Region {
    err: f32,
    q0: vec3<i32>,
    q1: vec3<i32>,
    idx: array<u32, 16>,
}

fn quantize3(h: vec3<f32>, bits: u32) -> vec3<i32> {
    let u = to_interp(h);
    return vec3<i32>(quantize(u.x, bits), quantize(u.y, bits), quantize(u.z, bits));
}

// Indices and error of region `s` for quantised endpoints.
fn evaluate(s: u32, q0: vec3<i32>, q1: vec3<i32>, bits: u32, ib: u32) -> Region {
    var r: Region;
    r.q0 = q0;
    r.q1 = q1;
    let a = vec3<i32>(unquantize(q0.x, bits), unquantize(q0.y, bits), unquantize(q0.z, bits));
    let b = vec3<i32>(unquantize(q1.x, bits), unquantize(q1.y, bits), unquantize(q1.z, bits));
    let fa = vec3<f32>(f32(finish(a.x)), f32(finish(a.y)), f32(finish(a.z)));
    let fb = vec3<f32>(f32(finish(b.x)), f32(finish(b.y)), f32(finish(b.z)));
    let d = fb - fa;
    let dd = dot(d, d);
    let top = (1u << ib) - 1u;
    var err = 0.0;
    for (var t = 0u; t < 16u; t++) {
        r.idx[t] = 0u;
        if (part[t] != s) {
            continue;
        }
        var k = 0;
        if (dd > 0.0) {
            k = i32(clamp(round(dot(px[t] - fa, d) / dd * f32(top)), 0.0, f32(top)));
        }
        var best_k = 0u;
        var best_e = 1e30;
        for (var o = -1; o <= 1; o++) {
            let cand = k + o;
            if (cand < 0 || cand > i32(top)) {
                continue;
            }
            let w = weight(ib, u32(cand));
            let v = (a * (64 - w) + b * w + vec3<i32>(32)) >> vec3<u32>(6u);
            let pal = vec3<f32>(f32(finish(v.x)), f32(finish(v.y)), f32(finish(v.z)));
            let e3 = pal - px[t];
            let e = dot(e3, e3);
            if (e < best_e) {
                best_e = e;
                best_k = u32(cand);
            }
        }
        r.idx[t] = best_k;
        err += best_e;
    }
    r.err = err;
    return r;
}

// Principal-axis endpoints of region `s` in the half-int domain: (e0, e1).
fn line_endpoints(s: u32) -> array<vec3<f32>, 2> {
    var mean = vec3<f32>(0.0);
    var n = 0.0;
    for (var t = 0u; t < 16u; t++) {
        if (part[t] == s) {
            mean += px[t];
            n += 1.0;
        }
    }
    mean /= max(n, 1.0);
    var cov = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var t = 0u; t < 16u; t++) {
        if (part[t] == s) {
            let d = px[t] - mean;
            cov += mat3x3<f32>(d * d.x, d * d.y, d * d.z);
        }
    }
    var v = cov[0];
    var largest = cov[0][0];
    for (var c = 1u; c < 3u; c++) {
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
    let axis = select(vec3<f32>(0.0), v / l, l > 1e-9);
    var tmin = 1e30;
    var tmax = -1e30;
    for (var t = 0u; t < 16u; t++) {
        if (part[t] == s) {
            let proj = dot(px[t] - mean, axis);
            tmin = min(tmin, proj);
            tmax = max(tmax, proj);
        }
    }
    return array<vec3<f32>, 2>(mean + axis * tmin, mean + axis * tmax);
}

fn fit_region(s: u32, bits: u32, ib: u32, iters: u32) -> Region {
    let e = line_endpoints(s);
    var best = evaluate(s, quantize3(e[0], bits), quantize3(e[1], bits), bits, ib);
    for (var i = 0u; i < iters; i++) {
        var aa = 0.0;
        var ab = 0.0;
        var bb = 0.0;
        var d0 = vec3<f32>(0.0);
        var d1 = vec3<f32>(0.0);
        for (var t = 0u; t < 16u; t++) {
            if (part[t] != s) {
                continue;
            }
            let w = f32(weight(ib, best.idx[t])) / 64.0;
            let iw = 1.0 - w;
            aa += iw * iw;
            ab += iw * w;
            bb += w * w;
            d0 += iw * px[t];
            d1 += w * px[t];
        }
        let det = aa * bb - ab * ab;
        if (abs(det) < 1e-6) {
            break;
        }
        let e0 = (bb * d0 - ab * d1) / det;
        let e1 = (aa * d1 - ab * d0) / det;
        let r = evaluate(s, quantize3(e0, bits), quantize3(e1, bits), bits, ib);
        if (r.err < best.err) {
            best = r;
        } else {
            break;
        }
    }
    return best;
}

struct Enc {
    err: f32,
    mode: u32,
    part: u32,
    fields: array<u32, 13>,
    idx: array<u32, 16>,
}

fn fits_signed(v: i32, bits: u32) -> bool {
    let half = 1 << (bits - 1u);
    return v >= -half && v < half;
}

// Mode `m` (index into the mode table) on partition `p`. `err` is 1e30 when a delta does not fit.
fn try_mode(m: u32, p: u32, iters: u32) -> Enc {
    let base = MODE_BASE + m * MODE_WORDS;
    let w0 = tables[base];
    let w1 = tables[base + 1u];
    let regions = (w0 >> 16u) & 0xFFu;
    let transformed = ((w0 >> 24u) & 0xFFu) == 1u;
    let epb = w1 & 0xFFu;
    let db = vec3<u32>((w1 >> 8u) & 0xFFu, (w1 >> 16u) & 0xFFu, (w1 >> 24u) & 0xFFu);
    let ib = select(4u, 3u, regions == 2u);
    set_partition(regions, p);
    var enc: Enc;
    enc.err = 0.0;
    enc.mode = m;
    enc.part = p;
    var q: array<vec3<i32>, 4>;
    for (var s = 0u; s < regions; s++) {
        let r = fit_region(s, epb, ib, iters);
        enc.err += r.err;
        var q0 = r.q0;
        var q1 = r.q1;
        var flip = false;
        for (var t = 0u; t < 16u; t++) {
            if (part[t] == s && is_anchor(regions, p, t) && ((r.idx[t] >> (ib - 1u)) & 1u) == 1u) {
                flip = true;
            }
        }
        if (flip) {
            q0 = r.q1;
            q1 = r.q0;
        }
        q[2u * s] = q0;
        q[2u * s + 1u] = q1;
        let top = (1u << ib) - 1u;
        for (var t = 0u; t < 16u; t++) {
            if (part[t] == s) {
                enc.idx[t] = select(r.idx[t], top - r.idx[t], flip);
            }
        }
    }
    let emask = (1u << epb) - 1u;
    for (var c = 0u; c < 3u; c++) {
        enc.fields[c] = u32(q[0][c]) & emask;
    }
    for (var i = 1u; i < 2u * regions; i++) {
        for (var c = 0u; c < 3u; c++) {
            if (transformed) {
                let d = q[i][c] - q[0][c];
                if (!fits_signed(d, db[c])) {
                    enc.err = 1e30;
                }
                enc.fields[3u * i + c] = u32(d) & ((1u << db[c]) - 1u);
            } else {
                enc.fields[3u * i + c] = u32(q[i][c]) & emask;
            }
        }
    }
    enc.fields[12] = p;
    return enc;
}

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

fn pack(e: Enc) -> vec4<u32> {
    let base = MODE_BASE + e.mode * MODE_WORDS;
    let w0 = tables[base];
    let regions = (w0 >> 16u) & 0xFFu;
    let len = tables[base + 2u];
    blk = array<u32, 4>(0u, 0u, 0u, 0u);
    bpos = 0u;
    put(w0 & 0xFFu, (w0 >> 8u) & 0xFFu);
    for (var b = 0u; b < len; b++) {
        let entry = (tables[base + 4u + b / 4u] >> (8u * (b % 4u))) & 0xFFu;
        let f = min(entry >> 4u, 12u);
        put((e.fields[f] >> (entry & 15u)) & 1u, 1u);
    }
    let ib = select(4u, 3u, regions == 2u);
    for (var t = 0u; t < 16u; t++) {
        put(e.idx[t], ib - select(0u, 1u, is_anchor(regions, e.part, t)));
    }
    return vec4<u32>(blk[0], blk[1], blk[2], blk[3]);
}

// The two-region partitions (of 32) with the smallest line-fit estimate.
fn best_partitions() -> vec2<u32> {
    var best = vec2<u32>(0u, 0u);
    var e1 = 1e30;
    var e2 = 1e30;
    for (var p = 0u; p < 32u; p++) {
        set_partition(2u, p);
        var err = 0.0;
        for (var s = 0u; s < 2u; s++) {
            let e = line_endpoints(s);
            let d = e[1] - e[0];
            let dd = dot(d, d);
            for (var t = 0u; t < 16u; t++) {
                if (part[t] == s) {
                    var k = 0.0;
                    if (dd > 0.0) {
                        k = round(clamp(dot(px[t] - e[0], d) / dd, 0.0, 1.0) * 7.0) / 7.0;
                    }
                    let r = px[t] - (e[0] + d * k);
                    err += dot(r, r);
                }
            }
        }
        if (err < e1) {
            e2 = e1;
            best.y = best.x;
            e1 = err;
            best.x = p;
        } else if (err < e2) {
            e2 = err;
            best.y = p;
        }
    }
    return best;
}

@compute @workgroup_size(8, 8)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.blocks.x || id.y >= params.blocks.y) {
        return;
    }
    for (var t = 0u; t < 16u; t++) {
        let x = min(id.x * 4u + (t & 3u), params.size.x - 1u);
        let y = min(id.y * 4u + (t >> 2u), params.size.y - 1u);
        let c = textureLoad(src, vec2<i32>(i32(x), i32(y)), 0).rgb;
        px[t] = vec3<f32>(half_int(c.x), half_int(c.y), half_int(c.z));
    }
    let iters = select(1u, 2u, QUALITY >= 2u);
    var best = try_mode(10u, 0u, iters);
    for (var m = 11u; m < 14u; m++) {
        if (best.err <= 0.0) {
            break;
        }
        let c = try_mode(m, 0u, iters);
        if (c.err < best.err) {
            best = c;
        }
    }
    if (QUALITY >= 1u && best.err > 0.0) {
        let parts = best_partitions();
        let tries = select(1u, 2u, QUALITY >= 2u);
        for (var k = 0u; k < tries; k++) {
            let p = select(parts.x, parts.y, k == 1u);
            for (var m = 0u; m < 10u; m++) {
                let c = try_mode(m, p, iters);
                if (c.err < best.err) {
                    best = c;
                }
            }
        }
    }
    out_blocks[params.out_offset + id.y * params.blocks.x + id.x] = pack(best);
}
