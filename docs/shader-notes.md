# Shader notes

Derivations and references for every shading model in the shader library (spec 0.3, 7.2). Each section
is referenced from the WGSL file that implements it; a shader with a model and no section here is a
defect.

Sections are added with the shaders they describe. Evidence for each library is in
`crates/shaders/tests/library.rs` (`[AC-WP05-2]`): the WGSL runs on the GPU through a compute harness
(`shaders/test/*.wgsl`) and is compared with a CPU reference or a closed form. Every shader of the
manifest is validated by naga for every flag set and override combination, then built as pipelines
on the device (`crates/shaders/tests/validate.rs`, `[AC-WP05-1]`).

## Shared math

`shaders/lib/math.wgsl`. Conventions as in `CLAUDE.md`: right-handed, Y up, column-major matrices
(`M * p`), quaternions `(x, y, z, w)`.

- **Quaternion product.** The Hamilton product of `a = (u, s)` and `b = (v, t)` is
  `(s v + t u + u × v, s t − u · v)`; `(a b) p` rotates by `b` first, as `humanart_math::Quat`.
- **Rotation of a vector.** For a unit `q = (u, w)`, `q v q*` expands to `v + 2w (u × v) + 2 u × (u × v)`.
  With `t = 2 (u × v)` this is `v + w t + u × t`: two cross products, no matrix (the form of
  `Quat::mul_vec3`).
- **Orthonormal basis.** Duff, Burgess, Christensen, Hery, Kensler, Liani, Villemin, "Building an
  Orthonormal Basis, Revisited", JCGT 6(1), 2017. With `s = sign(n.z)` (`+1` for `n.z = 0`),
  `a = −1 / (s + n.z)`, `b = n.x n.y a`:
  `t = (1 + s n.x² a, s b, −s n.x)`, `b = (b, s + n.y² a, −n.y)`. Since `|s + n.z| ≥ 1` there is no
  division by a small number; the frame is discontinuous only across `n.z = 0`. `t × b = n`
  (right-handed). The CPU twin `DVec3::any_orthonormal_pair` picks the sign with `copysign`, so the two
  differ only for `n.z = −0`.
- **safe_normalize** accepts a squared length in `(1e-30, f32::MAX]`: below, the direction is noise;
  above, the squared length overflowed to infinity.

Evidence: quaternion products and rotations equal `Quat` to 2e-6 (relative to `|v|`), the basis equals
the f64 construction to 2e-6, is orthonormal to 1e-5 and right-handed.

## Colour

`shaders/lib/color.wgsl`, `shaders/post/tonemap.wgsl`. Colours are linear Rec.709 (`[COL-02]`). Every
function has a CPU twin in `humanart_math::color` with the same constants; the tests parse the WGSL
constants and compare them with the CPU ones, so a change on one side fails until the other follows.

- **sRGB transfer.** IEC 61966-2-1: decode `v / 12.92` for `v ≤ 0.04045`, else
  `((v + 0.055) / 1.055)^2.4`; encode `12.92 v` for `v ≤ 0.0031308`, else `1.055 v^(1/2.4) − 0.055`.
  Both are extended as odd functions so negative (out-of-gamut) values survive a round trip.
- **Luminance.** `Y = 0.21263901 R + 0.71516868 G + 0.07219232 B`: row Y of the RGB-to-XYZ matrix built
  from the Rec.709 primaries and the D65 white `(0.3127, 0.3290)` (`LUMINANCE_REC709_F32`).
- **Exposure.** Lagarde and de Rousiers, "Moving Frostbite to Physically Based Rendering 3.0",
  SIGGRAPH 2014, section 4.5: the scale from scene luminance to the display range is
  `1 / (1.2 · 2^EV100)` (saturation-based sensitivity with the 78 % / 0.65 lens factor folded into
  1.2). EV100 is clamped to `[−32, 64]` (`EV100_RANGE`), so the scale stays a normal f32.
- **Khronos PBR Neutral** (default tone mapper, `TONE_MAPPER = 0`, name `pbr-neutral`). Khronos 3D
  Formats Working Group, "Khronos PBR Neutral Tone Mapper", 2024. With `x = min(r, g, b)`, the colour is
  lowered by `x − 6.25 x²` when `x < 0.08`, else by `0.04` (a smooth black offset that is 0 at 0 and
  0.04 above 0.08, matching the slope). With `p` the peak after that and `s = 0.76`: below `s` the colour
  passes; above, the new peak is `p′ = 1 − (1 − s)² / (p + (1 − s) − s)`, the colour is scaled by
  `p′ / p` and mixed towards the grey `p′` by `g = 1 − 1 / (0.15 (p − p′) + 1)`. Base colours up to 0.8
  are reproduced exactly (minus the offset), which is what product and skin renders want; hue is kept
  until the desaturation of very bright values.
- **ACES fitted** (`TONE_MAPPER = 1`, `aces-fitted`). Stephen Hill's fit (BakingLab `ACES.hlsl`, 2016) of
  the ACES RRT + sRGB ODT: input matrix (sRGB → AP1 with the RRT saturation), the rational curve
  `(v (v + 0.0245786) − 0.000090537) / (v (0.983729 v + 0.4329510) + 0.238081)` per component, output
  matrix (ODT saturation, AP1 → sRGB), clamp. The denominator has no real root. In f32, `v · v`
  overflows above about 1.8e19 and `inf / inf` would turn a highlight black, so the GPU clamps the
  curve input to ±1e6 (`ACES_FIT_INPUT_LIMIT`), where the curve is within 5e-7 of its asymptote
  `1 / 0.983729`; the f64 CPU version needs no clamp. (Found by the GPU tests: before the clamp an input
  of 1e30 mapped to black on the GPU and to white on the CPU.)
- **Clamp** (`TONE_MAPPER = 2`, `clamp`): each component to `[0, 1]`, NaN to 0; unknown mapper numbers
  also clamp. The numbering is the order of `ToneMapper::ALL` and the WGSL constant names are the
  stable names (`TONE_MAPPER_PBR_NEUTRAL`, ...), checked by the tests.
- **Display transform** (`post/tonemap.wgsl`): one full-screen triangle (vertices `(−1, −1)`, `(3, −1)`,
  `(−1, 3)`), `textureLoad` of the HDR texel, exposure, tone mapper, and the sRGB encode in the shader
  only when `ENCODE_SRGB` (a UNORM target; an `*Srgb` target encodes in hardware). Both are override
  constants: 6 pipelines, all built in the validation test.

Evidence (RTX 4060, Vulkan): over 20 008 colours from 1e-4 to 1e3 (and 0, 1e6, 1e30, a negative) at four
exposures, the worst |GPU − CPU| is 1.2e-7 for PBR Neutral and 4.2e-7 for ACES fitted; the sRGB curves
agree to 2e-6 relative, luminance to 1e-6, the exposure to 1e-6 relative.

## Packing

`shaders/lib/packing.wgsl`, CPU twin `humanart_math::pack` (`[EVAL-05]`).

- **Normalised integers.** snorm with `k` levels each side: `floor(0.5 + k · clamp(v, −1, 1))`, decoded
  `max(i / k, −1)` (so both minimum codes decode to −1); unorm with `k` levels:
  `floor(0.5 + k · clamp(v, 0, 1))`, decoded `i / k`. NaN quantises to 0 (tested on the bits, which
  survives fast-math). `10-10-10-2` words use 1023 / 511 levels for xyz and 3 / 1 for w, the
  `Rgb10a2Unorm` texel layout for the unorm variant.
- **Octahedral unit vectors.** Cigolle, Donow, Evangelakos, Mara, McGuire, Meyer, "A Survey of Efficient
  Representations for Independent Unit Vectors", JCGT 3(2), 2014. `n` is projected on the octahedron
  `|x| + |y| + |z| = 1`; the lower hemisphere is folded over the diagonals,
  `(x, y) → ((1 − |y|) sign(x), (1 − |x|) sign(y))`, with `sign(±0) = ±1` so the fold never collapses.
  The "precise" encoder of the survey quantises by trying the four grid points around the exact
  encoding and keeping the one whose decode is closest to `n`, compared by chord length (the dot
  product of two unit vectors is 1 in f32 at these angles and cannot rank them). Sizes: 2 × snorm16
  (`oct16`), 2 × snorm8 (`oct8`), and the tangent word: snorm16 x, a 15-bit snorm y and the bitangent
  sign in bit 31.
- **Halves.** `pack2x16float` pairs; `pack4x16float` is the layout of a `vec4<f16>`.

Evidence (RTX 4060, Vulkan, 20 003 directions and 20 012 value sets): `10-10-10-2` and half words are
bit-exact; all decoders agree to 2e-6 (halves bit-exact). The octahedral search resolves a near-tie
(two grid points equally close within f32 rounding; GPU `length` and division are not correctly
rounded) to the neighbouring code in 13 (oct16), 0 (oct8) and 6 (tangent) of 20 003 directions, with
an extra error of at most 1.6e-7 in chord length and never a worse code. The WGSL built-ins
`pack4x8snorm`, `pack4x8unorm`, `pack2x16snorm`, `pack2x16unorm` are the driver's: in 57 of 240 144
fields a value at (or within f32 rounding of) a half-way point rounds the other way, one code apart
(e.g. `pack4x8snorm(−0.5)` gives −64 on this driver, −63 by the WGSL formula). Data that must be
bit-identical between CPU and GPU uses the library encoders, not the built-ins.

## Sampling

`shaders/lib/sampling.wgsl`. Directions are local, +Z the normal.

- **Low-discrepancy points.** Hammersley: `(i / n, φ₂(i))` with `φ₂` the base-2 radical inverse
  (`reverseBits`). Roberts' R2: `frac(0.5 + i (1/g, 1/g²))` with `g ≈ 1.3247179572` the plastic number,
  computed in 0.32 fixed point (`i · 0xC13FA9A9 + 2^31`, `i · 0x91E10DA6 + 2^31`, wrapping), so it is
  exact for every u32 index. A 0.32 fraction converts to f32 through its top 24 bits times 2^-24, which
  is exact and stays below 1.
- **Noise.** Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH
  2014: interleaved gradient noise `frac(52.9829189 · frac(0.06711056 x + 0.00583715 y))`, the fallback
  when no blue-noise texture is bound; a blue-noise texel is decorrelated over frames by a
  Cranley-Patterson rotation with the R2 point of the frame.
- **Concentric disk.** Shirley and Chiu, "A Low Distortion Map Between Disk and Square", JGT 2(3), 1997.
  Area-preserving: `E[r²] = 1/2`.
- **Sphere and hemisphere.** `z = 1 − 2u` (sphere) or `z = u` (hemisphere), `φ = 2π u₂`: uniform by
  Archimedes' hat-box theorem; densities `1/(4π)` and `1/(2π)`.
- **Cosine hemisphere.** Malley's method: a uniform disk point lifted to the hemisphere has density
  `cos θ / π`. Moments: `E[z] = 2/3`, `E[z²] = 1/2`.
- **GGX NDF.** Walter, Marschner, Li, Torrance, "Microfacet Models for Refraction through Rough
  Surfaces", EGSR 2007: inverting the CDF of `D(h) cos θ_h` gives `cos² θ = (1 − u) / (1 + (α² − 1) u)`.
- **GGX VNDF.** Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT 7(4), 2018: stretch the
  view to the hemisphere configuration, sample the projected half-disk (warped by `s = (1 + v_h.z) / 2`),
  reproject and unstretch. Density `G1(v) max(v · h, 0) D(h) / v.z`.

Evidence (65 536 Hammersley points per warp): sequences bit-exact against the CPU; uniform sphere,
hemisphere, cosine and disk moments within 1e-3 of the closed forms; GGX NDF for α ∈ {0.2, 0.5, 0.9}:
the density equals f64 to 1e-4, `E[1/D] = ∫ cos θ dω = π` within 0.3 % (1/D is bounded, so the estimate
converges) and the sample moments match an f64 quadrature of `D cos θ`; GGX VNDF for the same widths
and view cosines 1, 0.6, 0.15: the reported density equals f64, integrates to 1 (quadrature, 2e-3) and
the sample moments `E[x], E[y], E[z], E[z²]` match its quadrature within 3e-3.

## BRDF

`shaders/lib/brdf.wgsl`, the microfacet building blocks of the skin model (`[SHD-01]`); `α` is the GGX
width, the perceptual roughness squared, never below 0.002 (below it `D` overflows f32 at `n·h = 1`).

- **Distribution.** Trowbridge-Reitz / GGX (Walter et al. 2007): `D(h) = α² / (π ((n·h)² (α² − 1) + 1)²)`,
  normalised so that `∫ D(h) (n·h) dω_h = 1`.
- **Masking.** Smith G1 of GGX: `2 n·v / (n·v + sqrt(α² + (1 − α²)(n·v)²))`. Height-correlated
  masking-shadowing (Heitz, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs",
  JCGT 3(2), 2014) folded with the `1 / (4 n·l n·v)` of the BRDF:
  `V = 0.5 / (n·l sqrt((n·v)²(1 − α²) + α²) + n·v sqrt((n·l)²(1 − α²) + α²))`, the exact form (not the
  approximation without the square roots).
- **Fresnel.** Schlick, "An Inexpensive BRDF Model for Physically-based Rendering", 1994:
  `F = f0 + (f90 − f0)(1 − cos)^5`. Exact unpolarised dielectric Fresnel (Walter et al. 2007, eq. 22)
  with `g² = η² − 1 + c²` (total internal reflection when negative). Skin `f0 = ((1.4 − 1)/(1.4 + 1))²
  ≈ 0.028`.
- **Dual lobe.** Two GGX lobes `D V` mixed by `blend` (`[SHD-01]`); the albedo is linear in the mix.
- **Split-sum DFG.** Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013: the directional albedo
  of one lobe is `f0 A + f90 B` with `A = ∫ (1 − Fc) f cos`, `B = ∫ Fc f cos`, `Fc = (1 − v·h)^5`,
  `f = D V`. Sampled with the VNDF (`pdf = G1 (v·h) D / (n·v) / (4 v·h)` in `l`), each sample weighs
  `f cos / pdf = 4 V n·l n·v / G1(v)`: `D` cancels, so the estimate has low variance.
- **Multi-scatter compensation.** Fdez-Agüera, "A Multiple-Scattering Microfacet Model for Real-Time
  Image-based Lighting", JCGT 8(1), 2019, in Filament's form: the single-scatter specular is scaled by
  `1 + f0 (1/E − 1)` with `E = A + B` (the albedo for `f0 = f90 = 1`).

Evidence (white furnace, midpoint rule on 1024 θ rows × 512 φ columns per cell, the rows on the GPU):
Lambert reflects 1 within 1e-5; `∫ D cos` is 1 within 2e-3 for α ∈ {0.1, 0.25, 0.5, 1}; the
single-scatter albedo with F = 1 never exceeds 1, equals an f64 CPU quadrature of the same integrand
within 2e-4 and, at α = 1 (where `D = 1/π` and `V = 1 / (2 (n·l + n·v))`), the closed form
`E = 1 − μ ln((1 + μ)/μ)` within 1e-4. Measured `E(n·v, α)`:

| n·v \ α | 0.1    | 0.25   | 0.5    | 1      |
|---------|--------|--------|--------|--------|
| 1       | 0.9883 | 0.9158 | 0.6878 | 0.3069 |
| 0.7     | 0.9814 | 0.8853 | 0.6830 | 0.3789 |
| 0.35    | 0.9479 | 0.8398 | 0.7292 | 0.5275 |
| 0.1     | 0.8830 | 0.8916 | 0.8601 | 0.7602 |

The loss grows with roughness away from grazing; at grazing view the visibility term makes it
non-monotonic (`E(0.1, 0.25) > E(0.1, 0.1)`), and rough lobes reflect more at grazing than at normal
incidence, as the α = 1 closed form shows. This energy loss is what the multi-scatter compensation
restores. The DFG kernel's `A + B` matches the furnace albedo within 4e-3 with 1024 and 4096 samples
and 3e-2 with 64, for all nine override combinations, and its results are bit-identical across the
three workgroup sizes. The exact Fresnel, Schlick, `f0` from the index and G1 agree with f64 to 1e-5
(1e-4 for incidence cosines below 1e-3).

## Skin pigment model

`[SKN-10]`. Implemented on the CPU in `crates/model/src/skin_appearance/pigment.rs`
(`PigmentModel`, `PigmentParams`, `Mapping`, `fit_lab`); the GPU skin compositor (`shaders/skin/`)
consumes its outputs. The model turns pigment parameters into a diffuse albedo, a subsurface-scattering
radius per colour channel and a reflectance spectrum. It is spectral, evaluated on 81 bands from 380 to
780 nm in 5 nm steps (`BANDS`, table in `crates/model/src/skin_appearance/spectra_table.rs`,
provenance in `docs/data-provenance.md`):

- **Epidermis** absorption (Donner & Jensen 2006, "A Spectral BSSRDF for Shading Human Skin", EGSR,
  eqs. 3 to 6), in mm⁻¹ with λ in nm: eumelanin `6.6e10 · λ^-3.33`, pheomelanin `2.9e14 · λ^-4.75`,
  baseline `0.0244 + 8.53 · exp(-(λ - 154) / 66.2)` (Jacques 1998, "Skin Optics", OMLC News), mixed as
  `Cm (βm σ_eu + (1 - βm) σ_pheo) + (1 - Cm) σ_base`, plus β-carotene `2.303 · ε_car · c`
  (PhotochemCAD molar extinction, c in mol/L).
- **Dermis** absorption (Donner & Jensen 2006, eq. 7): `Ch (γ σ_oxy + (1 - γ) σ_deoxy) + (1 - Ch)
  σ_base`, where the whole-blood coefficients come from Prahl's molar extinction table as
  `2.303 · ε · 150 g/L / 64 500 g/mol` (Prahl, OMLC, "Tabulated molar extinction coefficient for
  hemoglobin in water").
- **Melanin packaging**: eumelanin's absorption is the Duysens (1956) packet mean `(1 - e^{-σ s}) / s`
  of the Donner & Jensen coefficient, s = `Mapping::eumelanin_packet_mm` (0.035 mm, fitted to ISSA);
  pheomelanin is homogeneous.
- **Blood packaging**: blood sits in dermal vessels, so each haemoglobin species is packaged with the
  vessel factor `C = (1 - e^{-2 σ r}) / (2 σ r)` of Verkruysse et al. 1997 (Phys. Med. Biol. 42,
  51-65) as given by van Veen, Verkruysse & Sterenborg 2002 (Opt. Lett. 27, 246-248): the Duysens form
  with a path of 2r, r = `Mapping::blood_vessel_radius_mm` (0.013 mm, fitted to ISSA). Oxy and deoxy
  blood are packaged separately so the band table stays linear in the blood fraction and the
  oxygenation (no compositor shader change); packaging the mix exactly would absorb up to 11 % more in
  single bands, a colour change of at most CIEDE2000 0.34.
- **Scattering** (Donner & Jensen 2006, eq. 8): reduced scattering `14.74 · λ^-0.22 + 2.2e11 · λ^-4`
  mm⁻¹ in the epidermis, half of it in the dermis.
- **Reflectance**: the dermis is a semi-infinite medium with the total diffuse reflectance of the
  dipole diffusion model (Jensen & Buhler 2002, `Rd = α'/2 (1 + e^{-4/3 A √(3(1-α'))})
  e^{-√(3(1-α'))}`, η = 1.4 as in Donner & Jensen); the epidermis is an absorbing filter crossed twice
  (the round-trip path of Jacques 1998): `R = exp(-2 d σa_epi) · Rd`. Specular reflection at the
  surface is not part of the albedo (the BRDF owns it, `[SHD-01]`).
- **Colour**: `XYZ = Σ R S x̄ / Σ S ȳ` with the CIE 1931 2° observer and illuminant D65, then linear
  Rec.709 (no adaptation: D65 is the Rec.709 white).
- **SSS radius**: the dermis diffusion length `1 / σtr`, `σtr = √(3 σa (σa + σs'))`, averaged per
  channel with the positive part of the channel's spectral weight times `R(λ)`.

The registry stores every pigment parameter normalised to 0..1 (`data/template/params/skin.json`).
`PigmentParams::physical` maps them to physical quantities with the constants of `Mapping`. These
constants are **model choices, not measurements**: the blood range is Donner & Jensen's Table 1 range
(0.001 to 0.1), the epidermis range (0.03 to 0.09 mm) is centred on the 60 µm Jacques assumed when he
derived those fractions (so the default thickness reproduces his convention), and the carotene scale
(0 to 1e-5 mol/L) is uncalibrated. Three constants are fitted against ISSA: the melanin range (melanin
packet fraction 0 to 1, fitted with the packet path), the eumelanin packet path (0.035 mm) and the
blood vessel radius (0.013 mm, chosen on a grid of 0 to 0.02 mm to minimise the largest per-ITA-bin
spectral RMS with the colour tolerances kept). Both packet sizes are effective optical parameters,
not measured melanosome or vessel sizes.

Evidence and limits: the reference is ISSA (`data/test-scenes/skin-reference.json`: Lu et al. 2025,
Sci Data 12, 487, CC BY 4.0, 15 256 specular-included spectrophotometer records binned by ITA category,
L* and body site; ISSA records no Fitzpatrick type, so the six ITA categories stand in for I-VI).
`[AC-WP13-1]` is tested by `ac_wp13_1_reproduces_the_issa_reference_swatches` (every bin mean within
CIEDE2000 1.0, worst 0.81; the 36 decile-edge swatches within 2.5, worst 1.86) and
`ac_wp13_1_fitted_spectra_follow_the_measured_spectra` (spectral RMS 400-700 nm per ITA bin below
0.012, worst 0.0082), fitting melanin, eumelanin ratio and haemoglobin with the other parameters at
their registry defaults and the instrument's Fresnel surface reflection (n = 1.4) added. Blood
packaging took the light bins' spectral RMS from 0.0268 / 0.0308 (homogeneous blood) to 0.0072 / 0.0082
(`ac_wp13_1_blood_packaging_flattens_the_light_skin_spectra`). The tolerances were set after seeing the
results. `ac_wp13_1_fit_recovers_synthetic_targets` shows that `fit_lab` recovers synthetic Lab targets
within ΔE2000 < 0.5.

## Skin subsurface scattering

`[SHD-02]` to `[SHD-05]`, WP-14 (ADR-044). The CPU reference is `crates/render/src/sss.rs`
(`humanart_render::sss`); the GPU side is `shaders/render/geometry.wgsl` (skin class 1),
`shaders/render/subsurface.wgsl` (library) and `shaders/render/sss.wgsl` (the separable passes).
The tests use the CPU functions as the reference of what the shaders compute. Merged from the
module docs of `sss.rs` and the skin-shading agent's STATUS.

- **Profile.** The radial diffusion profile is the sum of six Gaussians that d'Eon, Luebke and
  Enderton fitted to the three-layer skin model (d'Eon & Luebke 2007, "Advanced Techniques for
  Realistic Real-Time Skin Rendering", GPU Gems 3 ch. 14, table 14-1): per channel
  `R(r) = Σ w_i G(v_i, r)` with `G(v, r) = exp(-r² / 2v) / (2π v)`, variances in mm²
  (`PROFILE_VARIANCES_MM2`, `PROFILE_WEIGHTS`). The weights of every channel sum to 1, so the
  profile only moves light: the albedo stays the albedo. A per-object width `scale` stretches the
  profile (`v_i → v_i scale²`); 1 is the measured skin. `SkinShading::scale_for_radius_mm` sets it
  from the mean of the pigment model's per-channel scattering radii (section "Skin pigment model");
  the per-channel shape stays the measured one (one width per pixel).
- **Separable kernel.** The 1D kernel is the profile's marginal along one axis,
  `k(x) = Σ w_i exp(-x² / 2v_i) / √(2π v_i)`; applied along x and then along y it keeps the energy
  and the per-axis variance of the profile (the separable approximation of Jimenez et al. 2015,
  "Separable Subsurface Scattering"). `KERNEL_TAPS` (25) taps at offsets that grow with the square
  of the tap index cover ±`KERNEL_RANGE_MM` (8.2 mm); each tap weighs the integral of `k` over its
  cell (computed with `erf`; the outer cells reach infinity), so every channel's weights sum to 1.
  On the GPU the kernel in mm is scaled by the per-pixel profile scale and the surface distance per
  pixel (from neighbour depths, so it shrinks at grazing angles). A tap on another surface (not
  skin, a depth step beyond the slope + 2 mm, or more than 3 mm x scale off the centre's tangent
  plane, fading over 3 mm) takes the centre value. One filtered fetch per tap: diffuse alpha holds
  the view depth, specular alpha the profile scale.
- **Transmittance.** Light entering the back of a slab `d` mm thick and leaving at the point
  opposite is the profile integrated over the back plane:
  `T(d) = ∫∫ R(√(r² + d²)) dA = Σ w_i exp(-d² / 2v_i)` (Jimenez et al. 2010, "Real-Time Realistic
  Skin Translucency"), 1 at `d = 0`; red reaches farthest. The thickness `d` comes from the shadow
  map of each shadowed directional or spot light, looked up 0.5 mm inside the surface, with wrap
  0.3; the baked thickness map `[BAKE-01]` is not used yet.
- **Pre-integrated.** For the Interactive tier the diffuse falloff of a light is integrated over a
  ring of radius `ρ` (Penner 2011, "Pre-Integrated Skin Shading"):
  `D(θ, ρ) = ∫ max(cos(θ + x), 0) R(2ρ sin(x/2)) dx / ∫ R(2ρ sin(x/2)) dx`, x over (-π, π],
  tabulated over `cos θ` and `scale / ρ` (32 x 32, `scale / ρ` up to 0.5 per mm); the curvature
  comes from `fwidth`. Interactive has no separable passes.
- **Specular.** Skin uses two GGX lobes (the material roughness and a secondary roughness 0.75,
  mixed 0.15), f0 0.028, multi-scatter compensation per lobe, and the diffuse is weighted by
  `1 -` the lobes' directional albedo.

Evidence (`crates/render/tests/skin.rs`, RTX 4060): skin furnace max 1.0000 in Review and
Interactive (bound < 1.015); GPU passes vs the CPU reference worst 0.13 % of peak; backlit
6 mm disc red 32.1 vs 32.8 from `T(5.5 mm)`; 4° past the terminator red 0.0037 of the lit side (Review), green and blue < 3e-5. Numbers in
`docs/perf-deviations.md`.
