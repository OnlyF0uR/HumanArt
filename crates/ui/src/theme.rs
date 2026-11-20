//! Themes and design tokens `[UI-08]`.
//!
//! A [`Theme`] is plain data: a [`Palette`] of named colours, [`Metrics`] (radii, spacing, control
//! sizes), a [`TypeScale`] of text styles, [`Shadows`] and [`Motion`]. Three built-in themes exist
//! ([`Theme::dark`], [`Theme::light`], [`Theme::high_contrast`]); every value can be overridden from
//! JSON ([`Theme::apply_json`]), which is how a debug build hot-reloads a theme file the application
//! watches. No colour or size is hard-coded in widget code: widgets read the active theme.
//!
//! The values come from the reference application's design tokens (`docs/reference-conventions.md`
//! section 16; its Black scheme is our dark theme, its White scheme our light theme), extended with
//! the tokens an editor needs (focus ring, warning, viewport surround, slider and switch sizes).
//! `docs/ui-design-tokens.md` documents every token.
//!
//! **Colour-neutral mode.** [`Theme::colour_neutral`] replaces every palette colour by the grey of
//! the same luminance and sets the viewport surround to an 18 % linear mid grey, so nothing around
//! the viewport tints the viewer's perception of skin colour.

use humanart_io::json::{self, Node, Object, Value};

use crate::color::Color;
use crate::error::UiError;

/// The font faces the UI ships (`data/fonts/`, all OFL `[LIC-04]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FontFace {
    /// Inter Regular: body text.
    Regular,
    /// Inter SemiBold: labels, titles, emphasis.
    SemiBold,
    /// Inter Italic.
    Italic,
    /// JetBrains Mono Regular: numbers in fields, code, the console.
    Mono,
    /// Lucide: icons, one private-use code point per icon.
    Icons,
}

impl FontFace {
    /// Every face, in the order of their index.
    pub const ALL: [Self; 5] = [Self::Regular, Self::SemiBold, Self::Italic, Self::Mono, Self::Icons];

    /// The face's index into [`FontFace::ALL`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The stable name used in theme files and baked font file names.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::SemiBold => "semibold",
            Self::Italic => "italic",
            Self::Mono => "mono",
            Self::Icons => "icons",
        }
    }
}

/// A text style: face, size in logical pixels and line height as a multiple of the size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle {
    /// The face.
    pub face: FontFace,
    /// The em size in logical pixels.
    pub size: f32,
    /// The line height as a multiple of `size`.
    pub line_height: f32,
}

impl TextStyle {
    /// A style.
    pub const fn new(face: FontFace, size: f32, line_height: f32) -> Self {
        Self { face, size, line_height }
    }

    /// The icon style of `size` (line height 1).
    pub const fn icon(size: f32) -> Self {
        Self::new(FontFace::Icons, size, 1.0)
    }

    /// The style with another size.
    #[must_use]
    pub const fn with_size(self, size: f32) -> Self {
        Self { size, ..self }
    }

    /// The style with another face.
    #[must_use]
    pub const fn with_face(self, face: FontFace) -> Self {
        Self { face, ..self }
    }
}

/// Defines a struct of named colours with a by-name view of its fields (for JSON and docs).
macro_rules! palette {
    ($($(#[doc = $doc:literal])* $name:ident,)*) => {
        /// Every colour the UI uses, by role.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Palette {
            $($(#[doc = $doc])* pub $name: Color,)*
        }

        impl Palette {
            /// The token names, in declaration order.
            pub const NAMES: &'static [&'static str] = &[$(stringify!($name),)*];

            /// The colours with their token names.
            pub fn entries(&self) -> impl Iterator<Item = (&'static str, Color)> + '_ {
                [$((stringify!($name), self.$name),)*].into_iter()
            }

            /// The colour named `name`, for overriding it.
            pub fn get_mut(&mut self, name: &str) -> Option<&mut Color> {
                match name {
                    $(stringify!($name) => Some(&mut self.$name),)*
                    _ => None,
                }
            }

            /// Every colour mapped through `f`.
            #[must_use]
            pub fn map(&self, mut f: impl FnMut(Color) -> Color) -> Self {
                Self { $($name: f(self.$name),)* }
            }

            /// Every colour blended towards `other` by `t` (theme cross-fades).
            #[must_use]
            pub fn mix(&self, other: &Self, t: f32) -> Self {
                Self { $($name: self.$name.mix(other.$name, t),)* }
            }
        }
    };
}

palette! {
    /// The main canvas behind panels' content.
    bg,
    /// Window chrome, docked panels; also the clear colour.
    panel,
    /// Raised surfaces: cards, menus, popovers, text fields.
    surface,
    /// A hovered element.
    hover,
    /// A selected or pressed element; slider and switch tracks.
    active,
    /// Hairline dividers.
    border,
    /// Borders of controls and floating surfaces.
    border_strong,
    /// The border of a focused input.
    border_focus,
    /// Primary text.
    text,
    /// Secondary text.
    text_muted,
    /// Hints, placeholders, disabled text.
    text_faint,
    /// The single accent colour (primary buttons, checked switches, slider fill).
    accent,
    /// Text and icons on the accent colour.
    on_accent,
    /// Text selection and selected rows.
    selection,
    /// The keyboard focus ring `[UI-09]`.
    focus_ring,
    /// Errors and destructive actions.
    danger,
    /// Success.
    positive,
    /// Warnings.
    warning,
    /// Links.
    link,
    /// Drop shadows under floating surfaces.
    shadow,
    /// The modal dimmer.
    scrim,
    /// Text on the scrim.
    on_scrim,
    /// The area around a viewport's image; always a hue-free grey.
    viewport_surround,
}

/// Defines a struct of named `f32` sizes with a by-name view (for JSON and docs).
macro_rules! metrics {
    ($($(#[doc = $doc:literal])* $name:ident,)*) => {
        /// Radii, spacing and control sizes, in logical pixels.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Metrics {
            $($(#[doc = $doc])* pub $name: f32,)*
        }

        impl Metrics {
            /// The token names, in declaration order.
            pub const NAMES: &'static [&'static str] = &[$(stringify!($name),)*];

            /// The sizes with their token names.
            pub fn entries(&self) -> impl Iterator<Item = (&'static str, f32)> + '_ {
                [$((stringify!($name), self.$name),)*].into_iter()
            }

            /// The size named `name`, for overriding it.
            pub fn get_mut(&mut self, name: &str) -> Option<&mut f32> {
                match name {
                    $(stringify!($name) => Some(&mut self.$name),)*
                    _ => None,
                }
            }
        }
    };
}

metrics! {
    /// Small radius: list rows, buttons, text fields.
    radius_sm,
    /// Default radius: menus, cards, tooltips.
    radius,
    /// Large radius: modals, large cards.
    radius_lg,
    /// Spacing step 1 (hairline gaps).
    space_1,
    /// Spacing step 2.
    space_2,
    /// Spacing step 3.
    space_3,
    /// Spacing step 4: the default gap between panels.
    space_4,
    /// Spacing step 5.
    space_5,
    /// Spacing step 6.
    space_6,
    /// Spacing step 7.
    space_7,
    /// Spacing step 8.
    space_8,
    /// Border width of controls and dividers.
    border,
    /// Width of the keyboard focus ring.
    focus_ring_width,
    /// Gap between a control and its focus ring.
    focus_ring_offset,
    /// Height of buttons, fields, sliders.
    control_height,
    /// Height of a compact control (toolbars, property rows).
    control_height_sm,
    /// Height of a list or tree row.
    row_height,
    /// Horizontal padding inside buttons and fields.
    control_pad_x,
    /// Icon size in buttons.
    icon,
    /// Icon size in compact controls.
    icon_sm,
    /// Width of an icon-only button (square).
    icon_button,
    /// Switch width.
    switch_w,
    /// Switch height.
    switch_h,
    /// Switch knob diameter.
    switch_knob,
    /// Slider track thickness.
    slider_track,
    /// Slider thumb diameter.
    slider_thumb,
    /// Width of the scroll bar's hit zone.
    scrollbar_zone,
    /// Scroll thumb thickness at rest.
    scrollbar_thin,
    /// Scroll thumb thickness while hovered or dragged.
    scrollbar_thick,
    /// Distance of the scroll thumb from the edge.
    scrollbar_inset,
    /// Shortest scroll thumb.
    scrollbar_min_thumb,
    /// Caret width in text fields.
    caret_width,
    /// Tooltip maximum width.
    tooltip_width,
    /// Tooltip padding.
    tooltip_pad,
}

/// The named text styles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TypeScale {
    /// Body text.
    pub body: TextStyle,
    /// Small text: property labels, list rows.
    pub small: TextStyle,
    /// Tiny text: captions under values, status bar.
    pub tiny: TextStyle,
    /// Control labels (buttons, tabs).
    pub label: TextStyle,
    /// Small caps-like section captions.
    pub caption: TextStyle,
    /// Panel headings.
    pub heading: TextStyle,
    /// Dialog titles.
    pub title: TextStyle,
    /// Numbers in fields, code, the console.
    pub mono: TextStyle,
}

impl TypeScale {
    /// The style names, in declaration order.
    pub const NAMES: &'static [&'static str] =
        &["body", "small", "tiny", "label", "caption", "heading", "title", "mono"];

    /// The styles with their names.
    pub fn entries(&self) -> [(&'static str, TextStyle); 8] {
        [
            ("body", self.body),
            ("small", self.small),
            ("tiny", self.tiny),
            ("label", self.label),
            ("caption", self.caption),
            ("heading", self.heading),
            ("title", self.title),
            ("mono", self.mono),
        ]
    }

    /// The style named `name`, for overriding it.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut TextStyle> {
        Some(match name {
            "body" => &mut self.body,
            "small" => &mut self.small,
            "tiny" => &mut self.tiny,
            "label" => &mut self.label,
            "caption" => &mut self.caption,
            "heading" => &mut self.heading,
            "title" => &mut self.title,
            "mono" => &mut self.mono,
            _ => return None,
        })
    }
}

/// A drop shadow recipe: draw the surface's shape blurred, offset down, before the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadow {
    /// Downward offset.
    pub offset_y: f32,
    /// Horizontal inset of the shadow shape at each side.
    pub inset_x: f32,
    /// Multiplier of the palette's `shadow` alpha.
    pub strength: f32,
    /// Corner radius of the shadow shape.
    pub radius: f32,
    /// Blur radius.
    pub blur: f32,
}

/// The shadow recipes by surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadows {
    /// Tooltips.
    pub tooltip: Shadow,
    /// Menus, dropdown lists.
    pub menu: Shadow,
    /// Popovers (colour picker, floating panels).
    pub popover: Shadow,
    /// Notices and toasts.
    pub notice: Shadow,
    /// Modal dialogs.
    pub modal: Shadow,
    /// Raised controls (slider thumb, switch knob).
    pub control: Shadow,
}

impl Shadows {
    /// The recipe names, in declaration order.
    pub const NAMES: &'static [&'static str] = &["tooltip", "menu", "popover", "notice", "modal", "control"];

    /// The recipes with their names.
    pub fn entries(&self) -> [(&'static str, Shadow); 6] {
        [
            ("tooltip", self.tooltip),
            ("menu", self.menu),
            ("popover", self.popover),
            ("notice", self.notice),
            ("modal", self.modal),
            ("control", self.control),
        ]
    }

    /// The recipe named `name`, for overriding it.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Shadow> {
        Some(match name {
            "tooltip" => &mut self.tooltip,
            "menu" => &mut self.menu,
            "popover" => &mut self.popover,
            "notice" => &mut self.notice,
            "modal" => &mut self.modal,
            "control" => &mut self.control,
            _ => return None,
        })
    }
}

/// Motion and timing tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    /// Rate of the standard exponential ease (hover, focus, toggles): `1 - exp(-dt * rate)` per frame.
    pub ease_rate: f32,
    /// Rate of the scroll ease.
    pub scroll_rate: f32,
    /// Duration of a theme cross-fade, seconds.
    pub theme_fade: f32,
    /// Largest frame time fed to animations, seconds.
    pub dt_clamp: f32,
    /// Hover time before a tooltip shows, seconds.
    pub tooltip_delay: f32,
    /// Caret blink period, seconds.
    pub caret_period: f32,
    /// Part of the period the caret is visible, seconds.
    pub caret_on: f32,
    /// Longest interval between clicks of a double click, seconds.
    pub double_click_time: f32,
    /// Largest distance between clicks of a double click, logical pixels.
    pub double_click_distance: f32,
    /// Distance a press must travel before it becomes a drag, logical pixels.
    pub drag_threshold: f32,
    /// One wheel line in logical pixels.
    pub wheel_line: f32,
    /// Reduced motion `[UI-09]`: eases jump to their target, the caret does not blink.
    pub reduced: bool,
}

/// Which built-in theme a theme started from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThemeKind {
    /// The default dark theme.
    Dark,
    /// The light theme.
    Light,
    /// The high-contrast theme `[UI-09]`.
    HighContrast,
}

impl ThemeKind {
    /// The stable name used in settings and theme files.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
            Self::HighContrast => "high-contrast",
        }
    }

    /// The kind with this name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "dark" => Some(Self::Dark),
            "light" => Some(Self::Light),
            "high-contrast" => Some(Self::HighContrast),
            _ => None,
        }
    }
}

/// A complete theme: plain data, cheap to copy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theme {
    /// The built-in theme this one started from.
    pub kind: ThemeKind,
    /// Colours.
    pub palette: Palette,
    /// Sizes.
    pub metrics: Metrics,
    /// Text styles.
    pub text: TypeScale,
    /// Shadow recipes.
    pub shadows: Shadows,
    /// Motion.
    pub motion: Motion,
    /// Whether the palette was neutralised by [`Theme::colour_neutral`].
    pub colour_neutral: bool,
}

/// The sizes every built-in theme shares.
const METRICS: Metrics = Metrics {
    radius_sm: 7.0,
    radius: 10.0,
    radius_lg: 16.0,
    space_1: 2.0,
    space_2: 4.0,
    space_3: 6.0,
    space_4: 8.0,
    space_5: 12.0,
    space_6: 16.0,
    space_7: 24.0,
    space_8: 32.0,
    border: 1.0,
    focus_ring_width: 2.0,
    focus_ring_offset: 2.0,
    control_height: 28.0,
    control_height_sm: 24.0,
    row_height: 30.0,
    control_pad_x: 12.0,
    icon: 15.0,
    icon_sm: 13.0,
    icon_button: 28.0,
    switch_w: 38.0,
    switch_h: 22.0,
    switch_knob: 16.0,
    slider_track: 4.0,
    slider_thumb: 14.0,
    scrollbar_zone: 12.0,
    scrollbar_thin: 3.0,
    scrollbar_thick: 6.0,
    scrollbar_inset: 4.0,
    scrollbar_min_thumb: 28.0,
    caret_width: 1.5,
    tooltip_width: 300.0,
    tooltip_pad: 10.0,
};

/// The type scale every built-in theme shares (reference section 16.3, sized for a dense editor).
const TYPE_SCALE: TypeScale = TypeScale {
    body: TextStyle::new(FontFace::Regular, 13.0, 1.5),
    small: TextStyle::new(FontFace::Regular, 12.0, 1.45),
    tiny: TextStyle::new(FontFace::Regular, 11.0, 1.4),
    label: TextStyle::new(FontFace::SemiBold, 12.5, 1.3),
    caption: TextStyle::new(FontFace::SemiBold, 11.0, 1.3),
    heading: TextStyle::new(FontFace::SemiBold, 14.0, 1.3),
    title: TextStyle::new(FontFace::SemiBold, 20.0, 1.25),
    mono: TextStyle::new(FontFace::Mono, 12.0, 1.5),
};

/// The shadow recipes every built-in theme shares (reference section 16.4).
const SHADOWS: Shadows = Shadows {
    tooltip: Shadow { offset_y: 6.0, inset_x: 0.0, strength: 1.0, radius: 10.0, blur: 20.0 },
    menu: Shadow { offset_y: 8.0, inset_x: 0.0, strength: 1.0, radius: 10.0, blur: 28.0 },
    popover: Shadow { offset_y: 8.0, inset_x: 0.0, strength: 1.0, radius: 12.0, blur: 22.0 },
    notice: Shadow { offset_y: 4.0, inset_x: 0.0, strength: 1.0, radius: 10.0, blur: 14.0 },
    modal: Shadow { offset_y: 16.0, inset_x: 0.0, strength: 1.0, radius: 16.0, blur: 48.0 },
    control: Shadow { offset_y: 1.0, inset_x: 0.0, strength: 0.5, radius: 8.0, blur: 3.0 },
};

/// The motion tokens every built-in theme shares (reference section 16.5).
const MOTION: Motion = Motion {
    ease_rate: 16.0,
    scroll_rate: 22.0,
    theme_fade: 0.25,
    dt_clamp: 0.05,
    tooltip_delay: 0.45,
    caret_period: 1.0,
    caret_on: 0.55,
    double_click_time: 0.4,
    double_click_distance: 4.0,
    drag_threshold: 3.0,
    wheel_line: 48.0,
    reduced: false,
};

impl Theme {
    /// The dark theme (the reference's Black scheme plus editor tokens).
    pub const fn dark() -> Self {
        Self {
            kind: ThemeKind::Dark,
            palette: Palette {
                bg: Color::hex(0x121212),
                panel: Color::hex(0x080808),
                surface: Color::hex(0x1A1A1A),
                hover: Color::hex(0x1F1F1F),
                active: Color::hex(0x2A2A2A),
                border: Color::hex(0x222222),
                border_strong: Color::hex(0x303030),
                border_focus: Color::hex(0x4C4C4C),
                text: Color::hex(0xEDEDED),
                text_muted: Color::hex(0xA3A3A3),
                text_faint: Color::hex(0x6B6B6B),
                accent: Color::hex(0xEDEDED),
                on_accent: Color::hex(0x0A0A0A),
                selection: Color::hexa(0xFFFFFF, 0.16),
                focus_ring: Color::hex(0x7DB3FF),
                danger: Color::hex(0xF26D6D),
                positive: Color::hex(0x6CCB8B),
                warning: Color::hex(0xE8B04A),
                link: Color::hex(0x7DB3FF),
                shadow: Color::hexa(0x000000, 0.6),
                scrim: Color::hexa(0x000000, 0.6),
                on_scrim: Color::hex(0xF5F5F5),
                viewport_surround: Color::hex(0x2B2B2B),
            },
            metrics: METRICS,
            text: TYPE_SCALE,
            shadows: SHADOWS,
            motion: MOTION,
            colour_neutral: false,
        }
    }

    /// The light theme (the reference's White scheme plus editor tokens).
    pub const fn light() -> Self {
        Self {
            kind: ThemeKind::Light,
            palette: Palette {
                bg: Color::hex(0xFFFFFF),
                panel: Color::hex(0xF4F4F5),
                surface: Color::hex(0xFFFFFF),
                hover: Color::hex(0xECECEE),
                active: Color::hex(0xE4E4E7),
                border: Color::hex(0xE6E6E9),
                border_strong: Color::hex(0xD6D6DB),
                border_focus: Color::hex(0xA1A1AA),
                text: Color::hex(0x18181B),
                text_muted: Color::hex(0x55555E),
                text_faint: Color::hex(0x9B9BA4),
                accent: Color::hex(0x18181B),
                on_accent: Color::hex(0xFAFAFA),
                selection: Color::hexa(0x2563EB, 0.16),
                focus_ring: Color::hex(0x2563EB),
                danger: Color::hex(0xDC2626),
                positive: Color::hex(0x15803D),
                warning: Color::hex(0xB45309),
                link: Color::hex(0x2563EB),
                shadow: Color::hexa(0x0F0F14, 0.09),
                scrim: Color::hexa(0x0C0C0F, 0.55),
                on_scrim: Color::hex(0xFFFFFF),
                viewport_surround: Color::hex(0xBDBDBD),
            },
            metrics: METRICS,
            text: TYPE_SCALE,
            shadows: SHADOWS,
            motion: MOTION,
            colour_neutral: false,
        }
    }

    /// The high-contrast theme `[UI-09]`: black surfaces, white text, strong borders, a yellow focus
    /// ring and wider rings and borders. Text on every surface exceeds a 7:1 contrast ratio.
    pub const fn high_contrast() -> Self {
        let mut metrics = METRICS;
        metrics.focus_ring_width = 3.0;
        metrics.caret_width = 2.0;
        metrics.scrollbar_thin = 5.0;
        metrics.scrollbar_thick = 8.0;
        Self {
            kind: ThemeKind::HighContrast,
            palette: Palette {
                bg: Color::hex(0x000000),
                panel: Color::hex(0x000000),
                surface: Color::hex(0x0A0A0A),
                hover: Color::hex(0x262626),
                active: Color::hex(0x3A3A3A),
                border: Color::hex(0x8A8A8A),
                border_strong: Color::hex(0xD0D0D0),
                border_focus: Color::hex(0xFFFFFF),
                text: Color::hex(0xFFFFFF),
                text_muted: Color::hex(0xE0E0E0),
                text_faint: Color::hex(0xB8B8B8),
                accent: Color::hex(0xFFFFFF),
                on_accent: Color::hex(0x000000),
                selection: Color::hexa(0xFFD400, 0.35),
                focus_ring: Color::hex(0xFFD400),
                danger: Color::hex(0xFF8080),
                positive: Color::hex(0x7CF29A),
                warning: Color::hex(0xFFD400),
                link: Color::hex(0x9CCBFF),
                shadow: Color::hexa(0x000000, 0.0),
                scrim: Color::hexa(0x000000, 0.8),
                on_scrim: Color::hex(0xFFFFFF),
                viewport_surround: Color::hex(0x3A3A3A),
            },
            metrics,
            text: TYPE_SCALE,
            shadows: SHADOWS,
            motion: MOTION,
            colour_neutral: false,
        }
    }

    /// The built-in theme of `kind`.
    pub const fn builtin(kind: ThemeKind) -> Self {
        match kind {
            ThemeKind::Dark => Self::dark(),
            ThemeKind::Light => Self::light(),
            ThemeKind::HighContrast => Self::high_contrast(),
        }
    }

    /// The colour-neutral variant `[UI-08]`: every colour becomes the grey of its luminance (alpha
    /// kept), and the viewport surround the 18 % linear mid grey. Idempotent.
    #[must_use]
    pub fn colour_neutral(&self) -> Self {
        let mut palette = self.palette.map(Color::neutral);
        palette.viewport_surround = Color::MID_GREY;
        Self { palette, colour_neutral: true, ..*self }
    }

    /// The theme with reduced motion switched on or off `[UI-09]`.
    #[must_use]
    pub fn with_reduced_motion(&self, reduced: bool) -> Self {
        let mut theme = *self;
        theme.motion.reduced = reduced;
        theme
    }

    /// A cross-fade between two themes' colours (sizes and styles come from `other` once `t` passes
    /// one half).
    #[must_use]
    pub fn mix(&self, other: &Self, t: f32) -> Self {
        let base = if t < 0.5 { self } else { other };
        Self { palette: self.palette.mix(&other.palette, t), ..*base }
    }

    /// Whether text is dark on a light background.
    pub fn is_light(&self) -> bool {
        self.palette.bg.luminance() > 0.5
    }

    /// Overrides values from a theme file `[UI-08]`.
    ///
    /// The file is a JSON object with any of the members `"base"` (`"dark"`, `"light"`,
    /// `"high-contrast"`: start from that built-in), `"colour_neutral"` (bool), `"palette"`
    /// (token → `"#RRGGBB"` or `"#RRGGBBAA"`, sRGB), `"metrics"` (token → number), `"text"` (style →
    /// `{ "face", "size", "line_height" }`), `"shadows"` (name → `{ "offset_y", "inset_x",
    /// "strength", "radius", "blur" }`) and `"motion"` (token → number or bool). Unknown names are
    /// errors, so a typo in a hot-reloaded file is reported instead of ignored. On error `self` is
    /// unchanged.
    pub fn apply_json(&mut self, text: &str) -> Result<(), UiError> {
        let value = json::parse_str(text).map_err(|e| UiError::ThemeFile(format!("{e}")))?;
        let mut theme = *self;
        read_theme(&value.node(), &mut theme).map_err(|e| UiError::ThemeFile(format!("{e}")))?;
        *self = theme;
        Ok(())
    }

    /// A theme from a theme file: `"base"` (default dark) with the file's overrides.
    pub fn from_json(text: &str) -> Result<Self, UiError> {
        let mut theme = Self::dark();
        theme.apply_json(text)?;
        Ok(theme)
    }

    /// The complete theme as a theme file (every token; [`Theme::from_json`] reads it back to an
    /// equal theme up to 8-bit colour quantisation).
    pub fn to_json(&self) -> Result<String, UiError> {
        let mut root = Object::new();
        root.insert("base", self.kind.name());
        root.insert("colour_neutral", self.colour_neutral);
        let mut palette = Object::new();
        let mut hex = String::new();
        for (name, color) in self.palette.entries() {
            hex.clear();
            color.write_hex(&mut hex);
            palette.insert(name, hex.as_str());
        }
        root.insert("palette", palette);
        let mut metrics = Object::new();
        for (name, v) in self.metrics.entries() {
            metrics.insert(name, f64::from(v));
        }
        root.insert("metrics", metrics);
        let mut text = Object::new();
        for (name, style) in self.text.entries() {
            let mut o = Object::new();
            o.insert("face", style.face.name());
            o.insert("size", f64::from(style.size));
            o.insert("line_height", f64::from(style.line_height));
            text.insert(name, o);
        }
        root.insert("text", text);
        let mut shadows = Object::new();
        for (name, s) in self.shadows.entries() {
            let mut o = Object::new();
            o.insert("offset_y", f64::from(s.offset_y));
            o.insert("inset_x", f64::from(s.inset_x));
            o.insert("strength", f64::from(s.strength));
            o.insert("radius", f64::from(s.radius));
            o.insert("blur", f64::from(s.blur));
            shadows.insert(name, o);
        }
        root.insert("shadows", shadows);
        let m = &self.motion;
        let mut motion = Object::new();
        for (name, v) in motion_numbers(m) {
            motion.insert(name, f64::from(v));
        }
        motion.insert("reduced", m.reduced);
        root.insert("motion", motion);
        json::to_string_pretty(&Value::from(root)).map_err(|e| UiError::ThemeFile(format!("{e}")))
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

/// The numeric motion tokens with their names.
fn motion_numbers(m: &Motion) -> [(&'static str, f32); 11] {
    [
        ("ease_rate", m.ease_rate),
        ("scroll_rate", m.scroll_rate),
        ("theme_fade", m.theme_fade),
        ("dt_clamp", m.dt_clamp),
        ("tooltip_delay", m.tooltip_delay),
        ("caret_period", m.caret_period),
        ("caret_on", m.caret_on),
        ("double_click_time", m.double_click_time),
        ("double_click_distance", m.double_click_distance),
        ("drag_threshold", m.drag_threshold),
        ("wheel_line", m.wheel_line),
    ]
}

/// The largest size any metric, style or shadow token may take, in logical pixels.
const MAX_TOKEN: f64 = 1000.0;

/// Reads a finite number in `0..=MAX_TOKEN`.
fn token_number(node: &Node<'_, '_>) -> Result<f32, json::AccessError> {
    let v = node.f64()?;
    if !(0.0..=MAX_TOKEN).contains(&v) {
        return Err(node.invalid("a token must be a number from 0 to 1000"));
    }
    Ok(v as f32)
}

/// Applies a parsed theme file to `theme`.
fn read_theme(root: &Node<'_, '_>, theme: &mut Theme) -> Result<(), json::AccessError> {
    root.deny_unknown(&["base", "colour_neutral", "palette", "metrics", "text", "shadows", "motion"])?;
    if let Some(base) = root.opt_field("base")? {
        let kind = base.choice(&[
            ("dark", ThemeKind::Dark),
            ("light", ThemeKind::Light),
            ("high-contrast", ThemeKind::HighContrast),
        ])?;
        *theme = Theme::builtin(kind);
    }
    if let Some(palette) = root.opt_field("palette")? {
        for (name, value) in palette.members()? {
            let Some(slot) = theme.palette.get_mut(name) else {
                return Err(value.invalid("unknown palette token"));
            };
            *slot = Color::parse_hex(value.str()?).ok_or_else(|| value.invalid("expected #RRGGBB or #RRGGBBAA"))?;
        }
    }
    if let Some(metrics) = root.opt_field("metrics")? {
        for (name, value) in metrics.members()? {
            let Some(slot) = theme.metrics.get_mut(name) else {
                return Err(value.invalid("unknown metric token"));
            };
            *slot = token_number(&value)?;
        }
    }
    if let Some(text) = root.opt_field("text")? {
        for (name, value) in text.members()? {
            let Some(slot) = theme.text.get_mut(name) else {
                return Err(value.invalid("unknown text style"));
            };
            value.deny_unknown(&["face", "size", "line_height"])?;
            if let Some(face) = value.opt_field("face")? {
                slot.face = face.choice(&[
                    ("regular", FontFace::Regular),
                    ("semibold", FontFace::SemiBold),
                    ("italic", FontFace::Italic),
                    ("mono", FontFace::Mono),
                    ("icons", FontFace::Icons),
                ])?;
            }
            if let Some(size) = value.opt_field("size")? {
                slot.size = token_number(&size)?.max(1.0);
            }
            if let Some(lh) = value.opt_field("line_height")? {
                slot.line_height = token_number(&lh)?.clamp(0.5, 4.0);
            }
        }
    }
    if let Some(shadows) = root.opt_field("shadows")? {
        for (name, value) in shadows.members()? {
            let Some(slot) = theme.shadows.get_mut(name) else {
                return Err(value.invalid("unknown shadow"));
            };
            value.deny_unknown(&["offset_y", "inset_x", "strength", "radius", "blur"])?;
            for (key, field) in [
                ("offset_y", &mut slot.offset_y),
                ("inset_x", &mut slot.inset_x),
                ("strength", &mut slot.strength),
                ("radius", &mut slot.radius),
                ("blur", &mut slot.blur),
            ] {
                if let Some(v) = value.opt_field(key)? {
                    *field = token_number(&v)?;
                }
            }
        }
    }
    if let Some(motion) = root.opt_field("motion")? {
        let m = &mut theme.motion;
        for (name, value) in motion.members()? {
            let slot = match name {
                "reduced" => {
                    m.reduced = value.bool()?;
                    continue;
                }
                "ease_rate" => &mut m.ease_rate,
                "scroll_rate" => &mut m.scroll_rate,
                "theme_fade" => &mut m.theme_fade,
                "dt_clamp" => &mut m.dt_clamp,
                "tooltip_delay" => &mut m.tooltip_delay,
                "caret_period" => &mut m.caret_period,
                "caret_on" => &mut m.caret_on,
                "double_click_time" => &mut m.double_click_time,
                "double_click_distance" => &mut m.double_click_distance,
                "drag_threshold" => &mut m.drag_threshold,
                "wheel_line" => &mut m.wheel_line,
                _ => return Err(value.invalid("unknown motion token")),
            };
            *slot = token_number(&value)?;
        }
    }
    if let Some(neutral) = root.opt_field("colour_neutral")?
        && neutral.bool()?
    {
        *theme = theme.colour_neutral();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_themes() -> [Theme; 3] {
        [Theme::dark(), Theme::light(), Theme::high_contrast()]
    }

    #[test]
    fn ui_08_text_contrast_on_surfaces() {
        // [UI-08] [UI-09] Primary text is readable on every surface of every theme; the
        // high-contrast theme reaches 7:1 (WCAG AAA) for primary and muted text.
        for theme in all_themes() {
            let p = theme.palette;
            for surface in [p.bg, p.panel, p.surface, p.hover] {
                assert!(p.text.contrast(surface) >= 7.0, "{:?}", theme.kind);
                assert!(p.text_muted.contrast(surface) >= 4.5, "{:?}", theme.kind);
            }
            assert!(p.on_accent.contrast(p.accent) >= 7.0);
            if theme.kind == ThemeKind::HighContrast {
                assert!(p.text_muted.contrast(p.surface) >= 7.0);
                assert!(p.focus_ring.contrast(p.bg) >= 7.0);
            }
        }
    }

    #[test]
    fn ui_08_colour_neutral_has_no_hue_and_mid_grey_surround() {
        for theme in all_themes() {
            let n = theme.colour_neutral();
            assert!(n.colour_neutral);
            for (name, c) in n.palette.entries() {
                assert!(c.r == c.g && c.g == c.b, "{name} keeps a hue");
            }
            assert_eq!(n.palette.viewport_surround, Color::MID_GREY);
            assert_eq!(n.colour_neutral(), n, "idempotent");
            // The surround of the normal themes is also hue-free.
            let s = theme.palette.viewport_surround;
            assert!(s.r == s.g && s.g == s.b);
        }
    }

    #[test]
    fn ui_08_json_round_trip() {
        for theme in all_themes() {
            for theme in [theme, theme.colour_neutral(), theme.with_reduced_motion(true)] {
                let text = theme.to_json().unwrap();
                let back = Theme::from_json(&text).unwrap();
                assert_eq!(back.kind, theme.kind);
                assert_eq!(back.metrics, theme.metrics);
                assert_eq!(back.text, theme.text);
                assert_eq!(back.shadows, theme.shadows);
                assert_eq!(back.motion, theme.motion);
                for ((name, a), (_, b)) in back.palette.entries().zip(theme.palette.entries()) {
                    assert_eq!(a.to_srgb8(), b.to_srgb8(), "{name}");
                }
            }
        }
    }

    #[test]
    fn ui_08_hot_reload_overrides_and_rejects_typos() {
        let mut theme = Theme::dark();
        theme
            .apply_json(
                r##"{"palette": {"accent": "#FF8800"}, "metrics": {"radius": 4}, "motion": {"reduced": true}}"##,
            )
            .unwrap();
        assert_eq!(theme.palette.accent.to_srgb8(), [0xFF, 0x88, 0x00, 0xFF]);
        assert_eq!(theme.metrics.radius, 4.0);
        assert!(theme.motion.reduced);
        let before = theme;
        for bad in [
            r##"{"palette": {"acent": "#FF8800"}}"##,
            r#"{"metrics": {"radius": -3}}"#,
            r#"{"metrics": {"radius": 1e9}}"#,
            r#"{"palette": {"accent": "orange"}}"#,
            r#"{"base": "sepia"}"#,
            r#"{"colours": {}}"#,
            r#"{"text": {"body": {"face": "comic"}}}"#,
            "not json",
        ] {
            assert!(theme.apply_json(bad).is_err(), "{bad}");
            assert_eq!(theme, before, "a failed reload leaves the theme unchanged");
        }
        let light = Theme::from_json(r#"{"base": "light"}"#).unwrap();
        assert_eq!(light, Theme::light());
    }

    #[test]
    fn mix_ends_are_the_themes() {
        let (a, b) = (Theme::dark(), Theme::light());
        assert_eq!(a.mix(&b, 0.0), a);
        assert_eq!(a.mix(&b, 1.0).palette, b.palette);
        assert!(!a.is_light() && b.is_light());
    }
}
