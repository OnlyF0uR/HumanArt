//! Wrinkle and tension drivers (section 6.2 step 6, `[EXP-04]`, `[EVAL-06]`).
//!
//! A driver belongs to one region and produces two activations in `[0, 1]` that the skin shader
//! uses to blend that region's wrinkle maps `[SKN-30]`:
//!
//! - **fold** (compression: deepen folds) `= clamp(Σ gain_e · e_e + compression_gain · compression_r, 0, 1)`,
//!   the expression weights `e_e` it listens to plus the measured mean compression of its region;
//! - **stretch** (reduce micro-detail contrast) `= clamp(stretch_gain · stretch_r, 0, 1)`.
//!
//! `compression_r` and `stretch_r` are the per-region reduction of [`super::strain::StrainField`]
//! (a region includes its subtree). Gains play the role of the region-wise strain normalisation of
//! `[BAKE-07]` until that baker exists.
//!
//! **Data file** (JSON):
//!
//! ```json
//! {"drivers": [
//!   {"name": "forehead.raise", "region": "forehead", "expressions": [["brow-raise", 1.0]],
//!    "compression": 8.0, "stretch": 6.0}
//! ]}
//! ```

use humanart_io::json::Value;

use super::error::{EvalError, EvalResult};
use crate::expression::ExpressionSet;
use crate::names::check_names;
use crate::region::RegionMap;

/// The most drivers one set may have.
pub const MAX_DRIVERS: usize = 4096;
/// The most expressions one driver may listen to.
pub const MAX_DRIVER_EXPRESSIONS: usize = 32;
/// The largest magnitude of a gain.
pub const MAX_GAIN: f32 = 1.0e3;

/// One driver for [`WrinkleDrivers::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct WrinkleDriver {
    /// Stable name (the shader's wrinkle-map slot is looked up by it).
    pub name: String,
    /// The region whose strain it reads.
    pub region: u16,
    /// `(expression index, gain)` pairs.
    pub expressions: Vec<(u32, f32)>,
    /// Gain on the region's mean compression.
    pub compression_gain: f32,
    /// Gain on the region's mean stretch.
    pub stretch_gain: f32,
}

/// A checked set of drivers in flat arrays.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WrinkleDrivers {
    names: Vec<String>,
    regions: Vec<u16>,
    offsets: Vec<u32>,
    expressions: Vec<u32>,
    gains: Vec<f32>,
    compression_gains: Vec<f32>,
    stretch_gains: Vec<f32>,
}

impl WrinkleDrivers {
    /// Checks `drivers` against `region_count` regions and `expression_count` expressions.
    pub fn new(drivers: &[WrinkleDriver], region_count: usize, expression_count: usize) -> EvalResult<Self> {
        if drivers.len() > MAX_DRIVERS {
            return Err(EvalError::limit("wrinkle driver count", MAX_DRIVERS, drivers.len()));
        }
        check_names("wrinkle driver", drivers.iter().map(|d| d.name.as_str()))?;
        let mut out = Self { offsets: vec![0], ..Self::default() };
        let bounded = |g: f32| g.is_finite() && g.abs() <= MAX_GAIN;
        for (i, d) in drivers.iter().enumerate() {
            if d.region as usize >= region_count {
                return Err(EvalError::Missing { what: "region", index: i });
            }
            if d.expressions.len() > MAX_DRIVER_EXPRESSIONS {
                return Err(EvalError::limit(
                    "wrinkle driver expressions",
                    MAX_DRIVER_EXPRESSIONS,
                    d.expressions.len(),
                ));
            }
            if d.expressions.iter().any(|&(e, _)| e as usize >= expression_count) {
                return Err(EvalError::Missing { what: "expression", index: i });
            }
            if !(d.expressions.iter().all(|&(_, g)| bounded(g))
                && bounded(d.compression_gain)
                && bounded(d.stretch_gain))
            {
                return Err(EvalError::Range {
                    what: "wrinkle driver",
                    index: i,
                    rule: "gain not finite or above MAX_GAIN",
                });
            }
            out.names.push(d.name.clone());
            out.regions.push(d.region);
            for &(e, g) in &d.expressions {
                out.expressions.push(e);
                out.gains.push(g);
            }
            out.offsets.push(out.expressions.len() as u32);
            out.compression_gains.push(d.compression_gain);
            out.stretch_gains.push(d.stretch_gain);
        }
        Ok(out)
    }

    /// Reads drivers from their JSON form (see the module comment), resolving names against the
    /// template's regions and expressions.
    pub fn from_json(value: &Value, regions: &RegionMap, expressions: &ExpressionSet) -> EvalResult<Self> {
        let root = value.node();
        root.deny_unknown(&["drivers"])?;
        let list = root.field("drivers")?;
        if list.array_len()? > MAX_DRIVERS {
            return Err(EvalError::limit("wrinkle driver count", MAX_DRIVERS, list.array_len()?));
        }
        let mut drivers = Vec::with_capacity(list.array_len()?);
        for d in list.elements()? {
            d.deny_unknown(&["name", "region", "expressions", "compression", "stretch"])?;
            let region_node = d.field("region")?;
            let region =
                regions.find(region_node.str()?).ok_or_else(|| region_node.invalid("no region of this name"))?;
            let mut pairs = Vec::new();
            if let Some(ex) = d.opt_field("expressions")? {
                if ex.array_len()? > MAX_DRIVER_EXPRESSIONS {
                    return Err(ex.invalid("too many expressions").into());
                }
                for item in ex.elements()? {
                    if item.array_len()? != 2 {
                        return Err(item.invalid("an expression input is [name, gain]").into());
                    }
                    let name_node = item.at(0)?;
                    let name = name_node.str()?;
                    let e = expressions
                        .expressions
                        .iter()
                        .position(|x| x.name == name)
                        .ok_or_else(|| name_node.invalid("no expression of this name"))?;
                    pairs.push((e as u32, item.at(1)?.f32()?));
                }
            }
            drivers.push(WrinkleDriver {
                name: d.field("name")?.str()?.to_owned(),
                region,
                expressions: pairs,
                compression_gain: d.get_or("compression", 0.0_f32)?,
                stretch_gain: d.get_or("stretch", 0.0_f32)?,
            });
        }
        Self::new(&drivers, regions.regions.len(), expressions.expressions.len())
    }

    /// How many drivers there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether there are no drivers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The driver names.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The region of every driver.
    #[must_use]
    pub fn regions(&self) -> &[u16] {
        &self.regions
    }

    /// Computes every driver's fold and stretch activation from expression weights and the
    /// per-region strain (missing entries count as 0) into `fold` and `stretch` (one entry per
    /// driver; extra entries are left alone). Non-finite inputs count as 0. Allocation-free.
    pub fn evaluate(
        &self,
        expression_weights: &[f32],
        region_stretch: &[f32],
        region_compression: &[f32],
        fold: &mut [f32],
        stretch: &mut [f32],
    ) -> EvalResult<()> {
        if fold.len() < self.len() || stretch.len() < self.len() {
            return Err(EvalError::Length {
                what: "wrinkle outputs (too few)",
                limit: self.len(),
                got: fold.len().min(stretch.len()),
            });
        }
        let read = |s: &[f32], i: usize| s.get(i).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
        for d in 0..self.len() {
            let r = self.regions[d] as usize;
            let range = self.offsets[d] as usize..self.offsets[d + 1] as usize;
            let mut f = self.compression_gains[d] * read(region_compression, r);
            for (&e, &g) in self.expressions[range.clone()].iter().zip(&self.gains[range]) {
                f += g * read(expression_weights, e as usize);
            }
            fold[d] = if f.is_finite() { f.clamp(0.0, 1.0) } else { 0.0 };
            let s = self.stretch_gains[d] * read(region_stretch, r);
            stretch[d] = if s.is_finite() { s.clamp(0.0, 1.0) } else { 0.0 };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::eval::evaluator::{EvalInput, Evaluator};
    use crate::eval::tests::face_patch;
    use humanart_io::json::parse_str;

    const DRIVERS: &str = r#"{"drivers": [
        {"name": "cheek-L.smile", "region": "face-L", "expressions": [["smile-L", 0.5]], "compression": 20, "stretch": 20},
        {"name": "face.brow", "region": "face", "expressions": [["brow-raise", 1.0]]}
    ]}"#;

    #[test]
    fn exp_04_drivers_follow_expressions_and_strain() {
        let t = Arc::new(face_patch(16, 0));
        let drivers = WrinkleDrivers::from_json(&parse_str(DRIVERS).unwrap(), t.regions(), t.expressions()).unwrap();
        assert_eq!(drivers.len(), 2);
        assert_eq!(drivers.regions(), &[1, 0]);
        assert_eq!(drivers.names()[1], "face.brow");
        let mut e = Evaluator::new(t.clone());
        let (mut fold, mut stretch) = ([0.0; 2], [0.0; 2]);
        let s = e.strain();
        drivers.evaluate(&[], s.region_stretch(), s.region_compression(), &mut fold, &mut stretch).unwrap();
        assert_eq!(fold, [0.0, 0.0]);
        assert_eq!(stretch, [0.0, 0.0]);
        let ex = [1.0, 0.0, 0.4, 0.0];
        e.evaluate(&EvalInput { expressions: &ex, ..EvalInput::default() }).unwrap();
        let s = e.strain();
        assert!(s.region_compression()[1] > 0.0 || s.region_stretch()[1] > 0.0, "the smile deforms the left region");
        drivers.evaluate(&ex, s.region_stretch(), s.region_compression(), &mut fold, &mut stretch).unwrap();
        assert!(fold[0] > 0.5 && fold[0] <= 1.0);
        assert!((fold[1] - 0.4).abs() < 1e-6);
        assert!(fold.iter().chain(&stretch).all(|v| (0.0..=1.0).contains(v)));
        assert!(drivers.evaluate(&ex, &[], &[], &mut fold[..1], &mut stretch).is_err());
    }

    #[test]
    fn bad_drivers_are_refused() {
        let t = face_patch(4, 0);
        let bad = [
            r#"{"drivers": [{"name": "a", "region": "nope"}]}"#,
            r#"{"drivers": [{"name": "a", "region": "face", "expressions": [["nope", 1]]}]}"#,
            r#"{"drivers": [{"name": "a", "region": "face", "compression": 1e9}]}"#,
            r#"{"drivers": [{"name": "a", "region": "face"}, {"name": "a", "region": "face"}]}"#,
            r#"{"drivers": [{"name": "a", "region": "face", "colour": 1}]}"#,
            r#"{"drivers": [{"name": "a", "region": "face", "expressions": [["smile-L"]]}]}"#,
        ];
        for (i, text) in bad.iter().enumerate() {
            let r = WrinkleDrivers::from_json(&parse_str(text).unwrap(), t.regions(), t.expressions());
            assert!(r.is_err(), "case {i} accepted");
        }
        let d = WrinkleDriver {
            name: "x".into(),
            region: 9,
            expressions: vec![],
            compression_gain: 0.0,
            stretch_gain: 0.0,
        };
        assert!(WrinkleDrivers::new(&[d], 3, 0).is_err());
        assert!(WrinkleDrivers::default().is_empty());
    }

    /// `[ROB-01]`: mutated driver files never panic or hang.
    #[test]
    fn driver_json_mutation_fuzz() {
        let t = face_patch(2, 0);
        let (regions, expressions) = (t.regions().clone(), t.expressions().clone());
        let corpus = vec![DRIVERS.as_bytes().to_vec()];
        humanart_core::fuzz::run(
            "wrinkle-drivers",
            &corpus,
            3000,
            0x0E04_F022,
            std::time::Duration::from_secs(60),
            move |bytes| {
                let Ok(v) = humanart_io::json::parse(bytes) else { return };
                if let Ok(d) = WrinkleDrivers::from_json(&v, &regions, &expressions) {
                    let (mut f, mut s) = (vec![0.0; d.len()], vec![0.0; d.len()]);
                    d.evaluate(&[1.0; 4], &[0.5; 3], &[0.5; 3], &mut f, &mut s).unwrap();
                    assert!(f.iter().chain(&s).all(|x| (0.0..=1.0).contains(x)));
                }
            },
        );
    }
}
