//! Colour science `[COL-03]`: the one shared, tested set of colour functions.
//!
//! The renderer's CPU references, the UI colour picker, the metrics of section 10.8, the reference-photo
//! calibration `[CMP-02]`, the light rig's colour temperatures `[LIT-01]` and the tone mappers whose
//! WGSL versions are tested against the CPU versions here `[COL-02]` all call into this module.
//!
//! Conventions: the working space is linear Rec.709/sRGB primaries with a D65 white `[COL-01]`.
//! Tristimulus values are plain arrays (`[f64; 3]`, and `[f32; 3]` where a GPU-side counterpart
//! exists); 3×3 matrices are [`matrix::ColorMatrix`], row-major (`m[row][column]`), and a colour
//! transforms as the column vector `M · c`. Hue angles are in degrees in `[0, 360)`.
//!
//! Every function is total. None panics on any input, NaN and infinities included (their results are
//! unspecified). Finite input gives finite output for every component magnitude up to `1e100` in
//! `f64` and `1e30` in `f32`, far beyond any radiometric value the engine handles; where a function
//! treats out-of-range input specially (clamping, saturation, a fallback), its documentation says how.
//!
//! # Module map
//!
//! | module | contents |
//! |---|---|
//! | [`matrix`] | 3×3 row-major matrix helpers (multiply, invert), `f32`/`f64` array conversion |
//! | [`transfer`] | sRGB / Display-P3 transfer functions (IEC 61966-2-1), 8-bit tables, generic gamma |
//! | [`spaces`] | RGB↔XYZ matrices derived from chromaticities (Rec.709, Display-P3, Rec.2020), Bradford adaptation, xyY, luminance |
//! | [`lab`] | CIELAB, LCh and Oklab |
//! | [`hsv`] | HSV and HSL for the colour picker |
//! | [`hex`] | `#RRGGBB` / `#RRGGBBAA` / `#RGB` parsing and formatting |
//! | [`delta_e`] | ΔE*76, ΔE*94 and CIEDE2000 (validated against Sharma, Wu & Dalal 2005) |
//! | [`temperature`] | Planckian and CIE daylight loci, colour temperature → linear RGB, McCamy CCT |
//! | [`tonemap`] | Khronos PBR Neutral and ACES fitted tone mappers, EV100 camera exposure |
//! | [`checker`] | 24-patch ColorChecker layout and reference, gain and 3×3 calibration fits |
//! | [`skin`] | Fitzpatrick swatch lookup over caller-supplied reference data, ITA°, skin-colour region, YCbCr and the vectorscope skin-tone line |
//! | [`stats`] | streaming per-region colour statistics in linear RGB and CIELAB |
//! | [`munsell`] | Lab / LCh readouts and the Munsell Value (ASTM D1535); Munsell hue and chroma need the renotation data |

pub mod checker;
pub mod delta_e;
pub mod hex;
pub mod hsv;
pub mod lab;
pub mod matrix;
pub mod munsell;
pub mod skin;
pub mod spaces;
pub mod stats;
pub mod temperature;
pub mod tonemap;
pub mod transfer;
