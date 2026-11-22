//! Expression presets and the range-of-motion sweep sequence (WP-19, `[EXP-07]`).
//!
//! **Data file** (`data/template/expressions/presets.json`, [`PRESETS_JSON`]):
//!
//! ```json
//! {"format": "humanart-expression-presets", "version": 1, "description": "...",
//!  "sweep": {"frames_per_control": 30},
//!  "presets": [{"name": "happiness.high", "group": "emotion",
//!               "controls": {"expression.au12_lip_corner_puller.l": 1.0}}]}
//! ```
//!
//! `group` is `neutral`, `emotion`, `viseme` or `custom`; control values are finite and in
//! `[0, 1]`; names follow the stable-name rules and are unique. Limits: at most [`MAX_PRESETS`]
//! presets, [`MAX_PRESET_CONTROLS`] controls per preset, `frames_per_control` in
//! `1..=`[`MAX_FRAMES_PER_CONTROL`], [`MAX_PRESETS_BYTES`] bytes. Unknown members are refused.
//! [`Presets::resolve`] checks the control names against a [`ControlRig`].
//!
//! **Range-of-motion sweep** (the quality-review diagnostic and the benchmark scene of
//! `[BENCH-01]`): [`Presets::sweep_frame`] writes the control vector of frame `f`: control
//! `f / (2 n)` alone (with `n` frames per control), rising linearly from 0 to 1 over `n` frames and
//! falling back over the next `n`. It allocates nothing.

use humanart_io::json::{Value, parse};

use super::controls::ControlRig;
use super::error::{ExprError, ExprResult};
use crate::names::check_name;

/// The checked-in presets of the template.
pub const PRESETS_JSON: &str = include_str!("../../../../data/template/expressions/presets.json");
/// The most presets.
pub const MAX_PRESETS: usize = 1024;
/// The most controls one preset sets.
pub const MAX_PRESET_CONTROLS: usize = 512;
/// The most frames per control of the sweep.
pub const MAX_FRAMES_PER_CONTROL: u32 = 10_000;
/// The most bytes of a presets file.
pub const MAX_PRESETS_BYTES: usize = 4 << 20;

/// What a preset is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresetGroup {
    /// The rest face.
    Neutral,
    /// A basic emotion at some intensity.
    Emotion,
    /// A speech viseme.
    Viseme,
    /// Anything else.
    Custom,
}

/// One named set of control values.
#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    /// Stable name (`happiness.high`, `viseme.aa`).
    pub name: String,
    /// What it is for.
    pub group: PresetGroup,
    /// Control names and values in `[0, 1]`; controls not listed are 0.
    pub controls: Vec<(String, f32)>,
}

/// The presets and sweep settings of a data file.
#[derive(Clone, Debug, PartialEq)]
pub struct Presets {
    /// The presets in file order.
    pub presets: Vec<Preset>,
    /// Frames per control and direction of the range-of-motion sweep.
    pub frames_per_control: u32,
}

impl Presets {
    /// Parses and checks a presets file (see the module comment).
    pub fn from_json_bytes(bytes: &[u8]) -> ExprResult<Self> {
        if bytes.len() > MAX_PRESETS_BYTES {
            return Err(ExprError::invalid("presets file", 0, "larger than 4 MiB"));
        }
        Self::from_json(&parse(bytes)?)
    }

    /// Reads the presets from a parsed file.
    pub fn from_json(value: &Value) -> ExprResult<Self> {
        let root = value.node();
        root.deny_unknown(&["format", "version", "description", "sweep", "presets"])?;
        if root.field("format")?.str()? != "humanart-expression-presets" {
            return Err(ExprError::invalid("presets file", 0, "format is not humanart-expression-presets"));
        }
        if root.field("version")?.u64()? != 1 {
            return Err(ExprError::invalid("presets file", 0, "unsupported version"));
        }
        let sweep = root.field("sweep")?;
        sweep.deny_unknown(&["frames_per_control"])?;
        let frames = sweep.field("frames_per_control")?.u64()?;
        if frames == 0 || frames > u64::from(MAX_FRAMES_PER_CONTROL) {
            return Err(ExprError::invalid("presets file", 0, "frames_per_control outside 1..=10000"));
        }
        let list = root.field("presets")?;
        if list.array_len()? > MAX_PRESETS {
            return Err(ExprError::invalid("presets file", 0, "too many presets"));
        }
        let mut presets: Vec<Preset> = Vec::with_capacity(list.array_len()?);
        for (i, p) in list.elements()?.enumerate() {
            p.deny_unknown(&["name", "group", "controls"])?;
            let name = p.field("name")?.str()?;
            check_name("expression preset", name)?;
            if presets.iter().any(|q| q.name == name) {
                return Err(ExprError::invalid("preset", i, "name used twice"));
            }
            let group = p.field("group")?.choice(&[
                ("neutral", PresetGroup::Neutral),
                ("emotion", PresetGroup::Emotion),
                ("viseme", PresetGroup::Viseme),
                ("custom", PresetGroup::Custom),
            ])?;
            let members = p.field("controls")?;
            let mut controls: Vec<(String, f32)> = Vec::new();
            for (k, (c, v)) in members.members()?.enumerate() {
                if k >= MAX_PRESET_CONTROLS {
                    return Err(ExprError::invalid("preset", i, "too many controls"));
                }
                check_name("expression control", c)?;
                let x = v.f32()?;
                if !(0.0..=1.0).contains(&x) {
                    return Err(ExprError::invalid("preset", i, "control value outside [0, 1]"));
                }
                if controls.iter().any(|(n, _)| n == c) {
                    return Err(ExprError::invalid("preset", i, "control given twice"));
                }
                controls.push((c.to_owned(), x));
            }
            presets.push(Preset { name: name.to_owned(), group, controls });
        }
        Ok(Self { presets, frames_per_control: frames as u32 })
    }

    /// The template's own presets ([`PRESETS_JSON`]).
    pub fn builtin() -> ExprResult<Self> {
        Self::from_json_bytes(PRESETS_JSON.as_bytes())
    }

    /// The preset called `name`.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&Preset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// Checks that every control named by a preset is a control of `rig`.
    pub fn resolve(&self, rig: &ControlRig) -> ExprResult<()> {
        for p in &self.presets {
            for (c, _) in &p.controls {
                if !rig.controls().iter().any(|x| x == c) {
                    return Err(ExprError::missing("rig control", c));
                }
            }
        }
        Ok(())
    }

    /// Writes preset `preset`'s control vector for `rig` into `out` (zeros elsewhere).
    pub fn apply(&self, preset: &Preset, rig: &ControlRig, out: &mut [f32]) -> ExprResult<()> {
        out.fill(0.0);
        for (c, v) in &preset.controls {
            let i = rig.controls().iter().position(|x| x == c).ok_or_else(|| ExprError::missing("rig control", c))?;
            if let Some(x) = out.get_mut(i) {
                *x = *v;
            }
        }
        Ok(())
    }

    /// How many frames the range-of-motion sweep over `controls` controls has.
    #[must_use]
    pub fn sweep_len(&self, controls: usize) -> u64 {
        2 * u64::from(self.frames_per_control) * controls as u64
    }

    /// Writes the controls of sweep frame `frame` into `out` (one control set, the rest 0) and
    /// returns that control's index; `None` (and all zeros) past the end. No allocation.
    pub fn sweep_frame(&self, frame: u64, out: &mut [f32]) -> Option<usize> {
        out.fill(0.0);
        let n = u64::from(self.frames_per_control.max(1));
        let c = usize::try_from(frame / (2 * n)).ok()?;
        let x = out.get_mut(c)?;
        let k = frame % (2 * n);
        // Up over frames 1..=n (frame n reaches 1), down over n+1..2n (back to 0 at the end).
        *x = if k <= n { k as f32 / n as f32 } else { (2 * n - k) as f32 / n as f32 };
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::fuzz;

    #[test]
    fn exp_07_builtin_presets() {
        let p = Presets::builtin().unwrap();
        assert!(p.find("neutral").is_some_and(|n| n.controls.is_empty()));
        for e in ["happiness", "sadness", "surprise", "fear", "anger", "disgust", "contempt"] {
            for i in ["low", "medium", "high"] {
                let preset = p.find(&format!("{e}.{i}")).unwrap();
                assert_eq!(preset.group, PresetGroup::Emotion);
                assert!(!preset.controls.is_empty());
            }
        }
        assert_eq!(p.presets.iter().filter(|x| x.group == PresetGroup::Viseme).count(), 15);
        // Contempt is one-sided (EMFACS R12A+R14A, todo 1.8): the character's right only.
        for i in ["low", "medium", "high"] {
            let c = &p.find(&format!("contempt.{i}")).unwrap().controls;
            assert_eq!(c.len(), 2, "{c:?}");
            assert!(c.iter().all(|(name, v)| name.rsplit('.').next() == Some("r") && *v > 0.0 && *v <= 1.0), "{c:?}");
        }
    }

    #[test]
    fn exp_07_sweep_rises_and_falls() {
        let p = Presets { presets: Vec::new(), frames_per_control: 4 };
        let mut out = [0.0_f32; 3];
        assert_eq!(p.sweep_len(3), 24);
        let mut seen = Vec::new();
        for f in 0..24 {
            let c = p.sweep_frame(f, &mut out).unwrap();
            assert!(out.iter().filter(|&&x| x != 0.0).count() <= 1);
            seen.push((c, out[c]));
        }
        assert_eq!(seen[0], (0, 0.0));
        assert_eq!(seen[4], (0, 1.0));
        assert_eq!(seen[6], (0, 0.5));
        assert_eq!(seen[8], (1, 0.0));
        assert_eq!(seen[12], (1, 1.0));
        assert_eq!(seen[23], (2, 0.25));
        assert_eq!(p.sweep_frame(24, &mut out), None);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn presets_files_are_checked() {
        let ok = r#"{"format":"humanart-expression-presets","version":1,"sweep":{"frames_per_control":2},"presets":[{"name":"a","group":"custom","controls":{"c":0.5}}]}"#;
        assert_eq!(Presets::from_json_bytes(ok.as_bytes()).unwrap().presets[0].controls, vec![("c".to_owned(), 0.5)]);
        for bad in [
            ok.replace("humanart-expression-presets", "x"),
            ok.replace("\"version\":1", "\"version\":2"),
            ok.replace("\"frames_per_control\":2", "\"frames_per_control\":0"),
            ok.replace("\"frames_per_control\":2", "\"frames_per_control\":20000"),
            ok.replace("0.5", "1.5"),
            ok.replace("0.5", "-0.1"),
            ok.replace("\"custom\"", "\"other\""),
            ok.replace("\"name\":\"a\"", "\"name\":\"a b\""),
            ok.replace("{\"c\":0.5}", "{\"c\":0.5,\"c\":0.2}"),
            ok.replace("\"group\"", "\"x\":1,\"group\""),
            ok.replace("}]}", "},{\"name\":\"a\",\"group\":\"custom\",\"controls\":{}}]}"),
        ] {
            assert!(Presets::from_json_bytes(bad.as_bytes()).is_err(), "{bad}");
        }
        assert!(Presets::from_json_bytes(&vec![b' '; MAX_PRESETS_BYTES + 1]).is_err());
    }

    #[test]
    fn rob_01_presets_file_mutation_fuzz() {
        let corpus = vec![PRESETS_JSON.as_bytes().to_vec()];
        fuzz::run("presets.json", &corpus, 2000, 0x19_0007, std::time::Duration::from_secs(30), |b| {
            if let Ok(p) = Presets::from_json_bytes(b) {
                let mut out = [0.0_f32; 4];
                let _ = p.sweep_frame(u64::MAX, &mut out);
            }
        });
    }
}
