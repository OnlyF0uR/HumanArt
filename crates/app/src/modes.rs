//! The editor's modes `[APP-02]`: what the Properties panel lists and which tools the mode offers.
//!
//! A mode names the registry groups its property grid shows (a group and its sub-groups,
//! [`humanart_doc::registry::Registry::in_group`]); the search box looks through the whole
//! registry whatever the mode, so every parameter is reachable from every mode (`[AC-WP21-1]`).
//! Ctrl+1 ... Ctrl+0 select the modes in order.

use humanart_ui::icons;

/// One editor mode `[APP-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Identity sliders, region sliders, landmark drag, sculpt.
    Shape,
    /// Skin layers, paint, procedural, generators.
    Skin,
    /// Eyes and mouth interior.
    EyesMouth,
    /// Hair, brows, lashes and facial hair.
    Hair,
    /// Expression controls, presets and sweep.
    Expression,
    /// The diagnostic pose.
    Pose,
    /// Lights and cameras.
    LightCamera,
    /// Comparison with references.
    Compare,
    /// The review rubric scenes.
    Review,
    /// Export.
    Export,
}

impl Mode {
    /// Every mode, in toolbar order.
    pub const ALL: [Self; 10] = [
        Self::Shape,
        Self::Skin,
        Self::EyesMouth,
        Self::Hair,
        Self::Expression,
        Self::Pose,
        Self::LightCamera,
        Self::Compare,
        Self::Review,
        Self::Export,
    ];

    /// The name shown in the mode selector.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Shape => "Shape",
            Self::Skin => "Skin",
            Self::EyesMouth => "Eyes & Mouth",
            Self::Hair => "Hair",
            Self::Expression => "Expression",
            Self::Pose => "Pose",
            Self::LightCamera => "Light & Camera",
            Self::Compare => "Compare",
            Self::Review => "Review",
            Self::Export => "Export",
        }
    }

    /// The stable id (keymap actions, saved workspaces).
    pub const fn id(self) -> &'static str {
        match self {
            Self::Shape => "mode.shape",
            Self::Skin => "mode.skin",
            Self::EyesMouth => "mode.eyes_mouth",
            Self::Hair => "mode.hair",
            Self::Expression => "mode.expression",
            Self::Pose => "mode.pose",
            Self::LightCamera => "mode.light_camera",
            Self::Compare => "mode.compare",
            Self::Review => "mode.review",
            Self::Export => "mode.export",
        }
    }

    /// The toolbar icon.
    pub const fn icon(self) -> char {
        match self {
            Self::Shape => icons::BOX,
            Self::Skin => icons::PALETTE,
            Self::EyesMouth => icons::EYE,
            Self::Hair => icons::ACTIVITY,
            Self::Expression => icons::SLIDERS,
            Self::Pose => icons::GRIP_VERTICAL,
            Self::LightCamera => icons::MAXIMIZE,
            Self::Compare => icons::COLUMNS,
            Self::Review => icons::CHECK_CIRCLE,
            Self::Export => icons::FILE,
        }
    }

    /// The registry groups the mode's property grid lists (each with its sub-groups).
    pub const fn groups(self) -> &'static [&'static str] {
        match self {
            Self::Shape => &["identity", "face", "body"],
            Self::Skin => &["skin"],
            Self::EyesMouth => &["eyes", "mouth"],
            Self::Hair => &["hair", "brows_lashes", "facial_hair"],
            Self::Expression => &["expression"],
            Self::Pose | Self::LightCamera | Self::Compare | Self::Review | Self::Export => &[],
        }
    }

    /// The mode of an id.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.id() == id)
    }

    /// Whether registry group `group` belongs to the mode.
    pub fn shows_group(self, group: &str) -> bool {
        self.groups()
            .iter()
            .any(|g| group == *g || (group.starts_with(g) && group.as_bytes().get(g.len()) == Some(&b'/')))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_02_every_registry_group_belongs_to_a_mode() {
        let reg = crate::tests::registry();
        for p in reg.params() {
            assert!(
                Mode::ALL.iter().any(|m| m.shows_group(&p.group)),
                "group {} of {} is in no mode's property grid",
                p.group,
                p.id
            );
        }
        for m in Mode::ALL {
            assert_eq!(Mode::from_id(m.id()), Some(m));
        }
        assert!(!Mode::Shape.shows_group("identityx"), "a prefix is not a sub-group");
    }
}
