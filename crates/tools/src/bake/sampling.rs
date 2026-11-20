//! The deterministic ray directions of the ray-traced bakers.
//!
//! Ray `i` of `n` at texel `(x, y)` uses the 2D point `(fract((i + ½)/n + r₁), fract(φ₂(i) + r₂))`,
//! where `φ₂` is the base-2 radical inverse (the Hammersley set) and `(r₁, r₂)` a Cranley-Patterson
//! rotation taken from a PCG hash of the texel and the seed. Every texel therefore gets a
//! well-stratified set, and neighbouring texels get different ones (noise instead of banding).
//! The point is warped to a cosine-weighted hemisphere or a uniform cone around a direction, in the
//! frame of [`Vec3::any_orthonormal_pair`] (Duff et al.).
//!
//! `shaders/bake/common.wgsl` implements the same functions with the same integer arithmetic; only
//! the trigonometry may differ in the last bits between the CPU and the GPU.

use humanart_math::Vec3;

/// The PCG hash (O'Neill's `pcg_hash`, the RXS-M-XS output of one LCG step).
pub const fn pcg_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// The hash of texel `(x, y)` for the bake seed `seed`.
pub const fn texel_hash(x: u32, y: u32, seed: u32) -> u32 {
    pcg_hash(x ^ pcg_hash(y ^ pcg_hash(seed)))
}

/// The Cranley-Patterson rotation `(r₁, r₂)` in `[0, 1)²` of a texel hash (16 bits each).
pub fn rotation(hash: u32) -> [f32; 2] {
    [(hash & 0xFFFF) as f32 / 65536.0, (hash >> 16) as f32 / 65536.0]
}

/// `x − floor(x)`, as WGSL's `fract`.
fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// The 2D point of ray `i` of `n` with rotation `r`.
pub fn point(i: u32, n: u32, r: [f32; 2]) -> [f32; 2] {
    let u1 = fract((i as f32 + 0.5) / n.max(1) as f32 + r[0]);
    let u2 = fract((i.reverse_bits() >> 8) as f32 * (1.0 / 16_777_216.0) + r[1]);
    [u1, u2]
}

/// A cosine-weighted direction around `+Z` for the point `u`.
pub fn cosine_hemisphere(u: [f32; 2]) -> Vec3 {
    let r = u[0].sqrt();
    let phi = std::f32::consts::TAU * u[1];
    Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - u[0]).max(0.0).sqrt())
}

/// A direction uniform in solid angle within the cone of half-angle `acos(cos_max)` around `+Z`.
pub fn uniform_cone(u: [f32; 2], cos_max: f32) -> Vec3 {
    let cos_theta = 1.0 - u[0] * (1.0 - cos_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = std::f32::consts::TAU * u[1];
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// The local direction `l` (around `+Z`) turned to the frame around the unit vector `n`.
pub fn to_world(l: Vec3, n: Vec3) -> Vec3 {
    let (t, b) = n.any_orthonormal_pair();
    t * l.x + b * l.y + n * l.z
}

/// `d` moved to the side of the plane with normal `g` that `toward` is on (`toward · g` gives the
/// side): a direction from an interpolated shading normal that points through the triangle's own
/// plane is mirrored back, so a ray never starts by crossing the surface it leaves.
pub fn keep_side(d: Vec3, g: Vec3, toward: f32) -> Vec3 {
    let s = d.dot(g);
    if s * toward < 0.0 { d - g * (2.0 * s) } else { d }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_matches_the_reference_values() {
        // Values of the reference C implementation.
        assert_eq!(pcg_hash(0), 129_708_002);
        assert_eq!(pcg_hash(1), 2_831_084_092);
        assert_ne!(texel_hash(3, 4, 0), texel_hash(4, 3, 0));
    }

    #[test]
    fn points_are_stratified_and_in_range() {
        let r = rotation(texel_hash(10, 20, 1));
        let n = 64;
        let mut cells = [0_u32; 8];
        for i in 0..n {
            let [a, b] = point(i, n, r);
            assert!((0.0..1.0).contains(&a) && (0.0..1.0).contains(&b));
            cells[(a * 8.0) as usize] += 1;
        }
        assert!(cells.iter().all(|&c| c == 8), "{cells:?}");
    }

    #[test]
    fn cosine_directions_have_the_cosine_mean() {
        // E[cos θ] = 2/3 for a cosine-weighted hemisphere.
        let n = 4096;
        let r = rotation(texel_hash(0, 0, 9));
        let mean: f32 = (0..n).map(|i| cosine_hemisphere(point(i, n, r)).z).sum::<f32>() / n as f32;
        assert!((mean - 2.0 / 3.0).abs() < 1e-3, "{mean}");
        let n_dir = Vec3::new(0.3, -0.5, 0.8).normalize();
        let d = to_world(cosine_hemisphere([0.0, 0.0]), n_dir);
        assert!((d - n_dir).length() < 1e-6);
    }

    #[test]
    fn cone_stays_in_the_cone_and_keep_side_mirrors() {
        let cos_max = 0.5;
        for i in 0..100 {
            let d = uniform_cone(point(i, 100, [0.3, 0.7]), cos_max);
            assert!(d.z >= cos_max - 1e-6 && (d.length() - 1.0).abs() < 1e-5);
        }
        let g = Vec3::Z;
        assert_eq!(keep_side(Vec3::new(1.0, 0.0, -1.0), g, 1.0), Vec3::new(1.0, 0.0, 1.0));
        assert_eq!(keep_side(Vec3::new(1.0, 0.0, 1.0), g, 1.0), Vec3::new(1.0, 0.0, 1.0));
    }
}
