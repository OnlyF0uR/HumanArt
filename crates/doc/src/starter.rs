//! Starter characters (todo 1.3): the parameter values a new character starts from, read from
//! `data/template/characters.json` (embedded, [`CHARACTERS_JSON`]).
//!
//! A starter names registry parameters and their values, and the scalp hair style a viewer shows
//! the character with (hair style is a view option, not a document parameter). [`DEFAULT`] is the
//! woman whose look the owner's reference images share; `average` keeps every registry default.
//! [`Starter::apply`] writes the values into a document's parameters, each checked against the
//! registry (an unknown id is an error, a value outside the parameter's range is clamped to it).
//!
//! The file is parsed under [`LIMITS`] and every count is checked before anything is stored
//! `[ROB-01]`.

use humanart_io::json::{JsonLimits, Value, parse_with};

use crate::document::Document;
use crate::error::DocError;
use crate::registry::{ParamKind, ParamValue, Registry};

/// The starter file shipped with the template.
pub const CHARACTERS_JSON: &str = include_str!("../../../data/template/characters.json");
/// The starter a new character uses when nothing else is asked for.
pub const DEFAULT: &str = "woman";
/// Most starters in a file.
pub const MAX_STARTERS: usize = 64;
/// Most parameter values in one starter.
pub const MAX_VALUES: usize = 1024;
/// Longest starter name or hair style, bytes.
pub const MAX_NAME_LEN: usize = 64;
/// The JSON limits of a starter file: 1 MiB, depth 8.
pub const LIMITS: JsonLimits = JsonLimits {
    max_depth: 8,
    max_input_bytes: 1 << 20,
    max_string_bytes: 1 << 16,
    max_members: 4096,
    max_values: 65_536,
};

/// One starter character.
#[derive(Clone, Debug, PartialEq)]
pub struct Starter {
    /// Stable name (`woman`, `man`, `average`).
    pub name: String,
    /// The scalp hair style a viewer shows it with (`short`, `medium-straight`, ...).
    pub hair: String,
    /// Registry ids and values, in file order.
    pub params: Vec<(String, f64)>,
}

/// The starters of a file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Starters {
    starters: Vec<Starter>,
}

fn bad(what: &str) -> DocError {
    DocError::Registry { id: "characters.json".into(), what: what.into() }
}

impl Starters {
    /// The starters shipped with the template ([`CHARACTERS_JSON`]).
    pub fn builtin() -> Result<Self, DocError> {
        Self::parse(CHARACTERS_JSON.as_bytes())
    }

    /// Parses a starter file.
    pub fn parse(bytes: &[u8]) -> Result<Self, DocError> {
        let value =
            parse_with(bytes, &LIMITS).map_err(|source| DocError::Json { file: "characters.json".into(), source })?;
        if value.get("format").and_then(Value::as_str) != Some("humanart-characters") {
            return Err(bad("not a humanart-characters file"));
        }
        let version = value.get("version").and_then(Value::as_u64).unwrap_or(0);
        if version != 1 {
            return Err(DocError::Version { format: "characters", found: version, supported: 1 });
        }
        let list = value.get("characters").and_then(Value::as_array).ok_or_else(|| bad("no characters array"))?;
        if list.len() > MAX_STARTERS {
            return Err(DocError::Limit { what: "starter characters", limit: MAX_STARTERS as u64 });
        }
        let mut starters: Vec<Starter> = Vec::with_capacity(list.len());
        for item in list {
            let text = |key: &str| -> Result<String, DocError> {
                let s = item.get(key).and_then(Value::as_str).ok_or_else(|| bad("a starter lacks a name or hair"))?;
                if s.is_empty() || s.len() > MAX_NAME_LEN {
                    return Err(bad("a starter's name or hair is empty or too long"));
                }
                Ok(s.to_owned())
            };
            let name = text("name")?;
            if starters.iter().any(|s| s.name == name) {
                return Err(bad("two starters share a name"));
            }
            let hair = text("hair")?;
            let members =
                item.get("params").and_then(Value::as_object).ok_or_else(|| bad("a starter has no params"))?;
            if members.entries().len() > MAX_VALUES {
                return Err(DocError::Limit { what: "values of a starter", limit: MAX_VALUES as u64 });
            }
            let mut params = Vec::with_capacity(members.entries().len());
            for (id, v) in members.entries() {
                let v = v.as_f64().filter(|v| v.is_finite()).ok_or_else(|| bad("a starter value is not a number"))?;
                params.push((id.clone(), v));
            }
            starters.push(Starter { name, hair, params });
        }
        Ok(Self { starters })
    }

    /// The starter named `name`.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&Starter> {
        self.starters.iter().find(|s| s.name == name)
    }

    /// Every starter, in file order.
    #[must_use]
    pub fn all(&self) -> &[Starter] {
        &self.starters
    }
}

impl Starter {
    /// Checks every value against `registry`: the id must name a scalar parameter that can be set.
    pub fn check(&self, registry: &Registry) -> Result<(), DocError> {
        for (id, _) in &self.params {
            let p = registry.get(id).ok_or_else(|| DocError::Registry {
                id: id.clone(),
                what: format!("starter `{}` sets a parameter the registry lacks", self.name),
            })?;
            if p.read_only || !matches!(p.kind, ParamKind::Scalar) {
                return Err(DocError::Registry {
                    id: id.clone(),
                    what: format!("starter `{}` sets a read-only or non-scalar parameter", self.name),
                });
            }
        }
        Ok(())
    }

    /// Writes the starter's values into `doc` ([`Starter::check`] first), each clamped to its
    /// parameter's range. A value equal to the parameter's default is not stored (a document keeps
    /// only values that differ from the registry), and removes a stored one.
    pub fn apply(&self, doc: &mut Document, registry: &Registry) -> Result<(), DocError> {
        self.check(registry)?;
        for (id, v) in &self.params {
            let Some(p) = registry.get(id) else { continue };
            let v = p.range.map_or(*v, |r| v.clamp(r.min, r.max));
            let value = ParamValue::Scalar(v);
            if value == p.default {
                doc.params.remove(id);
            } else {
                doc.params.insert(id.clone(), value);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;

    use super::*;

    #[test]
    fn todo_1_3_builtin_starters_parse_and_fit_the_registry() {
        let starters = Starters::builtin().unwrap();
        let names: Vec<&str> = starters.all().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["woman", "man", "average"]);
        let reg = crate::registry::tests::real_registry();
        for s in starters.all() {
            s.check(&reg).unwrap();
        }
        let woman = starters.find(DEFAULT).unwrap();
        assert_eq!(woman.hair, "medium-straight", "the default woman has longer hair");
        assert!(woman.params.contains(&("identity.sex".to_owned(), -1.0)));
        assert_eq!(starters.find("man").unwrap().hair, "short");
        assert!(starters.find("average").unwrap().params.is_empty());
        // Applying writes every value, clamped to its range.
        let me = crate::document::Author {
            id: "human.t".into(),
            kind: crate::document::AuthorKind::Human,
            name: "t".into(),
        };
        let mut doc = Document::new(humanart_core::id::Id128::from_u128(1), "t", "generated", 1, 0, me);
        woman.apply(&mut doc, &reg).unwrap();
        // Every value but `identity.sex` -1, the registry default, is stored.
        assert_eq!(doc.params.len(), woman.params.len() - 1);
        assert!(!doc.params.contains_key("identity.sex"));
        let wild = Starter { name: "x".into(), hair: "short".into(), params: vec![("identity.mass".into(), 99.0)] };
        wild.apply(&mut doc, &reg).unwrap();
        assert_eq!(doc.params.get("identity.mass"), Some(&ParamValue::Scalar(3.0)));
        let unknown = Starter { name: "x".into(), hair: "short".into(), params: vec![("no.such".into(), 1.0)] };
        assert!(unknown.apply(&mut doc, &reg).is_err());
    }

    #[test]
    fn rob_01_starter_files_survive_hostile_input() {
        for bad in [
            &b""[..],
            b"{}",
            br#"{"format":"humanart-characters","version":2,"characters":[]}"#,
            br#"{"format":"humanart-characters","version":1}"#,
            br#"{"format":"humanart-characters","version":1,"characters":[{"name":"a","hair":"short","params":{"x":"y"}}]}"#,
            br#"{"format":"humanart-characters","version":1,"characters":[{"name":"","hair":"short","params":{}}]}"#,
            br#"{"format":"humanart-characters","version":1,"characters":[{"name":"a","hair":"s","params":{}},{"name":"a","hair":"s","params":{}}]}"#,
        ] {
            assert!(Starters::parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        // Mutation fuzz: random byte flips, cuts and insertions never panic.
        let base = CHARACTERS_JSON.as_bytes();
        let mut rng = Rng::new(0x57a7_7e55);
        for _ in 0..2000 {
            let mut b = base.to_vec();
            for _ in 0..=(rng.next_u32() % 8) {
                let at = (rng.next_u32() as usize) % b.len().max(1);
                match rng.next_u32() % 3 {
                    0 => b[at] = rng.next_u32() as u8,
                    1 => b.truncate(at),
                    _ => b.insert(at, rng.next_u32() as u8),
                }
                if b.is_empty() {
                    break;
                }
            }
            let _ = Starters::parse(&b);
        }
    }
}
