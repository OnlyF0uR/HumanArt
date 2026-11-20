//! Landmark annotations of a scan `[WP-25]`: named points on the scan, named with the template's
//! stable landmark ids (`data/template/landmarks.json`, `humanart_model::landmark`), in the scan's
//! normalised frame (metres, Y up, facing +Z; [`super::scan::ScanMesh::normalise`]).
//!
//! The file format (`humanart-scan-landmarks`, version 1) is JSON:
//!
//! ```json
//! {"format": "humanart-scan-landmarks", "version": 1, "scan": "HSR0042-Body-045",
//!  "frame": "metres, Y up, facing +Z, after normalisation",
//!  "landmarks": [{"name": "nose.tip", "position": [0.0012, 1.6905, 0.1123]}]}
//! ```
//!
//! Hostile input `[ROB-01]`: the file is at most [`MAX_ANNOTATION_BYTES`], the JSON parser's own
//! limits apply, at most [`MAX_SCAN_LANDMARKS`] entries are read, names are at most
//! [`MAX_NAME_LEN`] bytes, unknown members are refused and every coordinate must be finite.

use std::fmt::Write as _;

use humanart_io::json::{self, Node};
use humanart_math::Vec3;
use humanart_model::Template;

use super::{CaptureError, Result};

/// The format name of an annotation file.
pub const FORMAT: &str = "humanart-scan-landmarks";
/// The format version written and read.
pub const VERSION: u64 = 1;
/// Largest annotation file.
pub const MAX_ANNOTATION_BYTES: usize = 1 << 20;
/// Most landmarks in one file.
pub const MAX_SCAN_LANDMARKS: usize = 4096;
/// Longest landmark or scan name, bytes.
pub const MAX_NAME_LEN: usize = 128;

/// One named point on a scan.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanLandmark {
    /// The template landmark id it corresponds to.
    pub name: String,
    /// Its position in the scan's normalised frame.
    pub position: Vec3,
}

/// The landmarks annotated on one scan.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LandmarkSet {
    /// The scan's name (free text).
    pub scan: String,
    /// The landmarks, names unique.
    pub landmarks: Vec<ScanLandmark>,
}

fn json_error(e: &impl std::fmt::Display) -> CaptureError {
    CaptureError::Annotation(e.to_string())
}

impl LandmarkSet {
    /// Parses an annotation file (see the module documentation).
    ///
    /// # Errors
    /// [`CaptureError::Limit`] for an oversized file or list; [`CaptureError::Annotation`] for
    /// malformed JSON, a wrong format or version, unknown members, a duplicate or over-long name
    /// or a non-finite coordinate.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_ANNOTATION_BYTES {
            return Err(CaptureError::limit("annotation file bytes", MAX_ANNOTATION_BYTES, bytes.len()));
        }
        let doc = json::parse(bytes).map_err(|e| json_error(&e))?;
        let root = Node::root(&doc);
        root.deny_unknown(&["format", "version", "scan", "frame", "landmarks"]).map_err(|e| json_error(&e))?;
        let format: &str = root.get("format").map_err(|e| json_error(&e))?;
        let version: u64 = root.get("version").map_err(|e| json_error(&e))?;
        if format != FORMAT || version != VERSION {
            return Err(CaptureError::Annotation(format!("not a {FORMAT} version {VERSION} file")));
        }
        let scan: String = root.get_or("scan", String::new()).map_err(|e| json_error(&e))?;
        if scan.len() > MAX_NAME_LEN {
            return Err(CaptureError::limit("scan name bytes", MAX_NAME_LEN, scan.len()));
        }
        let list = root.field("landmarks").map_err(|e| json_error(&e))?;
        let n = list.array_len().map_err(|e| json_error(&e))?;
        if n > MAX_SCAN_LANDMARKS {
            return Err(CaptureError::limit("annotation landmark count", MAX_SCAN_LANDMARKS, n));
        }
        let mut set = Self { scan, landmarks: Vec::with_capacity(n) };
        for item in list.elements().map_err(|e| json_error(&e))? {
            item.deny_unknown(&["name", "position"]).map_err(|e| json_error(&e))?;
            let name: String = item.get("name").map_err(|e| json_error(&e))?;
            let p: [f32; 3] = item.get("position").map_err(|e| json_error(&e))?;
            set.push(&name, Vec3::from_array(p))?;
        }
        Ok(set)
    }

    /// Adds a landmark.
    ///
    /// # Errors
    /// [`CaptureError::Annotation`] for an empty, over-long or duplicate name or a non-finite
    /// position; [`CaptureError::Limit`] beyond [`MAX_SCAN_LANDMARKS`].
    pub fn push(&mut self, name: &str, position: Vec3) -> Result<()> {
        if name.is_empty() || name.len() > MAX_NAME_LEN {
            return Err(CaptureError::Annotation(format!("landmark name '{name}' is empty or too long")));
        }
        if !position.is_finite() {
            return Err(CaptureError::Annotation(format!("landmark '{name}' has a non-finite position")));
        }
        if self.get(name).is_some() {
            return Err(CaptureError::Annotation(format!("landmark '{name}' is given twice")));
        }
        if self.landmarks.len() >= MAX_SCAN_LANDMARKS {
            return Err(CaptureError::limit("annotation landmark count", MAX_SCAN_LANDMARKS, self.landmarks.len() + 1));
        }
        self.landmarks.push(ScanLandmark { name: name.to_owned(), position });
        Ok(())
    }

    /// The position of landmark `name`.
    pub fn get(&self, name: &str) -> Option<Vec3> {
        self.landmarks.iter().find(|l| l.name == name).map(|l| l.position)
    }

    /// The annotation file text (format of the module documentation; 6 decimals, metres).
    pub fn to_json(&self) -> String {
        let mut out = format!(
            "{{\n  \"format\": \"{FORMAT}\",\n  \"version\": {VERSION},\n  \"scan\": {},\n  \"frame\": \"metres, Y up, facing +Z, after normalisation\",\n  \"landmarks\": [",
            quote(&self.scan)
        );
        for (i, l) in self.landmarks.iter().enumerate() {
            let sep = if i == 0 { "" } else { "," };
            let p = l.position;
            let _ = write!(
                out,
                "{sep}\n    {{\"name\": {}, \"position\": [{:.6}, {:.6}, {:.6}]}}",
                quote(&l.name),
                p.x,
                p.y,
                p.z
            );
        }
        out.push_str("\n  ]\n}\n");
        out
    }

    /// The pairs of this set with the template's landmarks of the same name: `(template landmark
    /// index, scan position)`, in this set's order, and the names the template does not have.
    pub fn match_template(&self, template: &Template) -> (Vec<(usize, Vec3)>, Vec<String>) {
        self.match_names(&template.landmarks().iter().map(|l| l.name.as_str()))
    }

    /// [`LandmarkSet::match_template`] against a list of landmark names (index = position in
    /// `names`).
    pub fn match_names<'a, I: Iterator<Item = &'a str> + Clone>(&self, names: &I) -> (Vec<(usize, Vec3)>, Vec<String>) {
        let mut pairs = Vec::with_capacity(self.landmarks.len());
        let mut unknown = Vec::new();
        for l in &self.landmarks {
            match names.clone().position(|t| t == l.name) {
                Some(i) => pairs.push((i, l.position)),
                None => unknown.push(l.name.clone()),
            }
        }
        (pairs, unknown)
    }
}

/// `text` as a JSON string literal.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    json::write_string(&mut out, text);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp25_annotation_round_trip() {
        let mut set = LandmarkSet { scan: "test \"scan\"".into(), landmarks: Vec::new() };
        set.push("nose.tip", Vec3::new(0.001, 1.69, 0.112)).unwrap();
        set.push("chin.menton", Vec3::new(0.0, 1.55, 0.09)).unwrap();
        let text = set.to_json();
        let back = LandmarkSet::parse(text.as_bytes()).unwrap();
        assert_eq!(back.scan, set.scan);
        assert_eq!(back.landmarks.len(), 2);
        assert!(back.get("nose.tip").unwrap().abs_diff_eq(Vec3::new(0.001, 1.69, 0.112), 1e-6));
        assert!(set.push("nose.tip", Vec3::ZERO).is_err());
        assert!(set.push("x", Vec3::new(f32::NAN, 0.0, 0.0)).is_err());
        assert!(set.push("", Vec3::ZERO).is_err());
    }

    #[test]
    fn wp25_annotations_match_template_names() {
        let names = ["head.vertex", "nose.tip"];
        let mut set = LandmarkSet::default();
        set.push("nose.tip", Vec3::ZERO).unwrap();
        set.push("no.such.landmark", Vec3::ZERO).unwrap();
        let (pairs, unknown) = set.match_names(&names.iter().copied());
        assert_eq!(pairs, vec![(1, Vec3::ZERO)]);
        assert_eq!(unknown, vec!["no.such.landmark".to_owned()]);
    }

    #[test]
    fn wp25_bad_annotation_files_fail_cleanly() {
        for bad in [
            &b"{}"[..],
            br#"{"format": "other", "version": 1, "landmarks": []}"#,
            br#"{"format": "humanart-scan-landmarks", "version": 2, "landmarks": []}"#,
            br#"{"format": "humanart-scan-landmarks", "version": 1, "landmarks": [{"name": "a", "position": [1, 2]}]}"#,
            br#"{"format": "humanart-scan-landmarks", "version": 1, "landmarks": [{"name": "a", "position": [1, 2, 3], "x": 1}]}"#,
            br#"{"format": "humanart-scan-landmarks", "version": 1, "landmarks": [{"name": "a", "position": [1, 2, 1e999]}]}"#,
        ] {
            assert!(LandmarkSet::parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        assert!(LandmarkSet::parse(&vec![b' '; MAX_ANNOTATION_BYTES + 1]).is_err());
        // [ROB-01] Mutations never panic or hang.
        let mut set = LandmarkSet::default();
        set.push("nose.tip", Vec3::new(0.0, 1.7, 0.1)).unwrap();
        set.push("L.eye.canthus.medial", Vec3::new(0.016, 1.73, 0.09)).unwrap();
        let corpus = vec![set.to_json().into_bytes()];
        humanart_core::fuzz::run("scan-landmarks", &corpus, 5000, 0x1a2b, std::time::Duration::from_secs(60), |b| {
            if let Ok(s) = LandmarkSet::parse(b) {
                assert!(s.landmarks.iter().all(|l| l.position.is_finite()));
            }
        });
    }
}
