//! Code points of the icon face (Lucide, `data/fonts/icons.anfont`) used by the widgets, by their
//! Lucide names. The values were read from the font's `post` glyph names and `cmap` (the same font
//! and code points as the reference application's `icon` module).

/// `chevron-down`.
pub const CHEVRON_DOWN: char = '\u{e06d}';
/// `chevron-left`.
pub const CHEVRON_LEFT: char = '\u{e06e}';
/// `chevron-right`.
pub const CHEVRON_RIGHT: char = '\u{e06f}';
/// `chevron-up`.
pub const CHEVRON_UP: char = '\u{e070}';
/// `check`.
pub const CHECK: char = '\u{e06c}';
/// `x`.
pub const X: char = '\u{e1b2}';
/// `search`.
pub const SEARCH: char = '\u{e151}';
/// `lock`.
pub const LOCK: char = '\u{e10b}';
/// `lock-open`.
pub const LOCK_OPEN: char = '\u{e10c}';
/// `link`.
pub const LINK: char = '\u{e102}';
/// `unlink`.
pub const UNLINK: char = '\u{e19c}';
/// `pipette` (eyedropper).
pub const PIPETTE: char = '\u{e13b}';
/// `grip-vertical`.
pub const GRIP_VERTICAL: char = '\u{e0eb}';
/// `minus`.
pub const MINUS: char = '\u{e11c}';
/// `plus`.
pub const PLUS: char = '\u{e13d}';
/// `info`.
pub const INFO: char = '\u{e0f9}';
/// `alert-circle`.
pub const ALERT_CIRCLE: char = '\u{e077}';
/// `alert-triangle`.
pub const ALERT_TRIANGLE: char = '\u{e193}';
/// `check-circle`.
pub const CHECK_CIRCLE: char = '\u{e07c}';
/// `keyboard`.
pub const KEYBOARD: char = '\u{e284}';
/// `maximize-2`.
pub const MAXIMIZE: char = '\u{e113}';
/// `minimize-2`.
pub const MINIMIZE: char = '\u{e11b}';
/// `pin`.
pub const PIN: char = '\u{e259}';
/// `rotate-ccw` (reset).
pub const RESET: char = '\u{e148}';
/// `zoom-in`.
pub const ZOOM_IN: char = '\u{e1b6}';
/// `zoom-out`.
pub const ZOOM_OUT: char = '\u{e1b7}';
/// `scan` (fit to view).
pub const FIT: char = '\u{e257}';
/// `terminal`.
pub const TERMINAL: char = '\u{e181}';
/// `settings`.
pub const SETTINGS: char = '\u{e154}';
/// `play`.
pub const PLAY: char = '\u{e13c}';
/// `pause`.
pub const PAUSE: char = '\u{e12e}';
/// `eye`.
pub const EYE: char = '\u{e0ba}';
/// `copy`.
pub const COPY: char = '\u{e09e}';
/// `trash-2`.
pub const TRASH: char = '\u{e18e}';
/// `ellipsis`.
pub const MORE: char = '\u{e0b6}';
/// `app-window` (a floating panel).
pub const WINDOW: char = '\u{e426}';
/// `columns-2` (split side by side).
pub const COLUMNS: char = '\u{e098}';
/// `rows-2` (split one above the other).
pub const ROWS: char = '\u{e439}';
/// `palette`.
pub const PALETTE: char = '\u{e1dd}';
/// `sliders-horizontal`.
pub const SLIDERS: char = '\u{e29a}';
/// `list-tree`.
pub const TREE: char = '\u{e408}';
/// `folder`.
pub const FOLDER: char = '\u{e0d7}';
/// `file`.
pub const FILE: char = '\u{e0c0}';
/// `image`.
pub const IMAGE: char = '\u{e0f6}';
/// `box`.
pub const BOX: char = '\u{e061}';
/// `activity`.
pub const ACTIVITY: char = '\u{e038}';
/// `bell`.
pub const BELL: char = '\u{e059}';

/// Every icon above, for tests and the shortcut overlay's legend.
pub const ALL: &[char] = &[
    CHEVRON_DOWN,
    CHEVRON_LEFT,
    CHEVRON_RIGHT,
    CHEVRON_UP,
    CHECK,
    X,
    SEARCH,
    LOCK,
    LOCK_OPEN,
    LINK,
    UNLINK,
    PIPETTE,
    GRIP_VERTICAL,
    MINUS,
    PLUS,
    INFO,
    ALERT_CIRCLE,
    ALERT_TRIANGLE,
    CHECK_CIRCLE,
    KEYBOARD,
    MAXIMIZE,
    MINIMIZE,
    PIN,
    RESET,
    ZOOM_IN,
    ZOOM_OUT,
    FIT,
    TERMINAL,
    SETTINGS,
    PLAY,
    PAUSE,
    EYE,
    COPY,
    TRASH,
    MORE,
    WINDOW,
    COLUMNS,
    ROWS,
    PALETTE,
    SLIDERS,
    TREE,
    FOLDER,
    FILE,
    IMAGE,
    BOX,
    ACTIVITY,
    BELL,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fonts;
    use crate::theme::FontFace;

    #[test]
    fn every_icon_is_in_the_baked_icon_face() {
        let icons = fonts().face(FontFace::Icons);
        for &c in ALL {
            assert!(icons.has_glyph(c), "U+{:04X} is not in icons.anfont", u32::from(c));
            assert!(!icons.glyph(icons.glyph_index(c)).is_blank(), "U+{:04X} is blank", u32::from(c));
        }
    }
}
