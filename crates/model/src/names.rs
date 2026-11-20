//! Stable names and the body side, shared by every table of the template.
//!
//! Regions, landmarks, joints, UV tiles, mesh parts, delta components and expressions are identified
//! by a **stable name**: once published, a name is never renamed or reused for something else
//! (CLAUDE.md). The index of an entry in its table is package-local and may change between template
//! versions; documents and agents refer to entries by name.
//!
//! A name is 1 to [`MAX_NAME_LEN`] bytes of ASCII letters, digits, `.`, `-` and `_`, starting with a
//! letter or digit. That covers the spec's schemes (`L.eye.upper.04`, `brow-L`, `jaw.L.02`).

use crate::error::{ModelError, Result};

/// The longest stable name, in bytes.
pub const MAX_NAME_LEN: usize = 64;

/// Checks that `name` follows the naming rules; `table` names the table in the error.
pub fn check_name(table: &'static str, name: &str) -> Result<()> {
    let what = if name.is_empty() {
        Some("is empty")
    } else if name.len() > MAX_NAME_LEN {
        Some("is longer than 64 bytes")
    } else if !name.as_bytes()[0].is_ascii_alphanumeric() {
        Some("does not start with a letter or digit")
    } else if !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_')) {
        Some("has a character other than a letter, digit, '.', '-' or '_'")
    } else {
        None
    };
    match what {
        None => Ok(()),
        Some(what) => Err(name_error(table, name, what)),
    }
}

/// Checks every name of a table and that no name occurs twice.
pub fn check_names<'a>(table: &'static str, names: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut sorted: Vec<&str> = Vec::new();
    for name in names {
        check_name(table, name)?;
        sorted.push(name);
    }
    sorted.sort_unstable();
    for pair in sorted.windows(2) {
        if pair[0] == pair[1] {
            return Err(name_error(table, pair[0], "is used twice"));
        }
    }
    Ok(())
}

/// The position of `name` in `names`, if present.
pub fn find_name<'a>(mut names: impl Iterator<Item = &'a str>, name: &str) -> Option<usize> {
    names.position(|n| n == name)
}

fn name_error(table: &'static str, name: &str, what: &'static str) -> ModelError {
    let mut cut = name.len().min(MAX_NAME_LEN);
    while !name.is_char_boundary(cut) {
        cut -= 1;
    }
    ModelError::Name { table, name: name[..cut].to_owned(), what }
}

/// Which side of the body an entry belongs to. Left is the character's left (+X, since the
/// character faces +Z in a right-handed Y-up frame).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Side {
    /// On the midline, or not sided.
    #[default]
    Center,
    /// The character's left (+X).
    Left,
    /// The character's right (-X).
    Right,
}

impl Side {
    /// The byte stored in packages.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Center => 0,
            Self::Left => 1,
            Self::Right => 2,
        }
    }

    /// The side for a stored byte, if valid.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Center),
            1 => Some(Self::Left),
            2 => Some(Self::Right),
            _ => None,
        }
    }

    /// The other side; the centre stays the centre.
    #[must_use]
    pub const fn mirror(self) -> Self {
        match self {
            Self::Center => Self::Center,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_rules() {
        for good in ["L.eye.upper.04", "brow-L", "jaw.L.02", "a", "0_x", &"n".repeat(64)] {
            assert!(check_name("t", good).is_ok(), "{good}");
        }
        for bad in ["", ".x", "-x", "a b", "é", "a/b", &"n".repeat(65)] {
            assert!(check_name("t", bad).is_err(), "{bad}");
        }
        assert!(check_names("t", ["a", "b", "c"].into_iter()).is_ok());
        let dup = check_names("region", ["a", "b", "a"].into_iter());
        assert!(matches!(dup, Err(ModelError::Name { table: "region", what: "is used twice", .. })));
        assert_eq!(find_name(["a", "b"].into_iter(), "b"), Some(1));
        assert_eq!(find_name(["a", "b"].into_iter(), "c"), None);
    }

    #[test]
    fn long_non_ascii_names_are_cut_on_a_char_boundary() {
        let name = "é".repeat(40);
        match check_name("t", &name) {
            Err(ModelError::Name { name, .. }) => assert!(name.len() <= MAX_NAME_LEN),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn sides_round_trip() {
        for s in [Side::Center, Side::Left, Side::Right] {
            assert_eq!(Side::from_u8(s.to_u8()), Some(s));
            assert_eq!(s.mirror().mirror(), s);
        }
        assert_eq!(Side::from_u8(3), None);
    }
}
