//! Input events and the rebindable keymap `[UI-06]`.
//!
//! The windowing layer (the app shell) translates its events into [`Event`]s, in logical pixels,
//! and pushes them into an [`InputQueue`]; [`crate::ui::Ui::begin_frame`] consumes the queue. The
//! types are independent of winit so the toolkit stays testable and wasm-safe: tests drive the UI
//! by pushing synthetic events.
//!
//! The clipboard is handled by the shell: on paste it pushes [`Event::Paste`] with the clipboard's
//! text, and it writes [`crate::ui::FrameOutput::copied`] to the clipboard after a frame.

use std::fmt;

use humanart_io::json::{self, Object, Value};

use crate::error::UiError;
use crate::geom::Point;

/// A key, independent of the keyboard layout where it matters for shortcuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// Tab.
    Tab,
    /// Enter or Return.
    Enter,
    /// Escape.
    Escape,
    /// The space bar.
    Space,
    /// Backspace.
    Backspace,
    /// Delete (forward).
    Delete,
    /// Left arrow.
    Left,
    /// Right arrow.
    Right,
    /// Up arrow.
    Up,
    /// Down arrow.
    Down,
    /// Home.
    Home,
    /// End.
    End,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Insert.
    Insert,
    /// A letter key, upper-case `'A'..='Z'`.
    Letter(char),
    /// A digit key `'0'..='9'`.
    Digit(char),
    /// A function key, 1 to 24.
    F(u8),
    /// Any other key with a printable name (`"Minus"`, `"BracketLeft"`).
    Named(&'static str),
}

impl Key {
    /// The stable name used in keymap files (`"Tab"`, `"A"`, `"5"`, `"F2"`, `"Minus"`).
    pub fn write_name(self, out: &mut String) {
        let name = match self {
            Self::Tab => "Tab",
            Self::Enter => "Enter",
            Self::Escape => "Escape",
            Self::Space => "Space",
            Self::Backspace => "Backspace",
            Self::Delete => "Delete",
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Up => "Up",
            Self::Down => "Down",
            Self::Home => "Home",
            Self::End => "End",
            Self::PageUp => "PageUp",
            Self::PageDown => "PageDown",
            Self::Insert => "Insert",
            Self::Letter(c) | Self::Digit(c) => {
                out.push(c);
                return;
            }
            Self::F(n) => {
                out.push('F');
                out.push_str(&n.to_string());
                return;
            }
            Self::Named(n) => n,
        };
        out.push_str(name);
    }

    /// The key with this name (see [`Key::write_name`]); named keys must be listed in `named`.
    pub fn from_name(name: &str, named: &[&'static str]) -> Option<Self> {
        let key = match name {
            "Tab" => Self::Tab,
            "Enter" => Self::Enter,
            "Escape" => Self::Escape,
            "Space" => Self::Space,
            "Backspace" => Self::Backspace,
            "Delete" => Self::Delete,
            "Left" => Self::Left,
            "Right" => Self::Right,
            "Up" => Self::Up,
            "Down" => Self::Down,
            "Home" => Self::Home,
            "End" => Self::End,
            "PageUp" => Self::PageUp,
            "PageDown" => Self::PageDown,
            "Insert" => Self::Insert,
            _ => {
                let mut chars = name.chars();
                return match (chars.next(), chars.next()) {
                    (Some(c), None) if c.is_ascii_uppercase() => Some(Self::Letter(c)),
                    (Some(c), None) if c.is_ascii_digit() => Some(Self::Digit(c)),
                    (Some('F'), Some(_)) => name[1..].parse::<u8>().ok().filter(|n| (1..=24).contains(n)).map(Self::F),
                    _ => named.iter().find(|n| **n == name).map(|n| Self::Named(n)),
                };
            }
        };
        Some(key)
    }
}

impl fmt::Display for Key {
    /// The stable name of [`Key::write_name`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Letter(c) | Self::Digit(c) => write!(f, "{c}"),
            Self::F(n) => write!(f, "F{n}"),
            Self::Named(n) => f.write_str(n),
            other => {
                let name = match other {
                    Self::Tab => "Tab",
                    Self::Enter => "Enter",
                    Self::Escape => "Escape",
                    Self::Space => "Space",
                    Self::Backspace => "Backspace",
                    Self::Delete => "Delete",
                    Self::Left => "Left",
                    Self::Right => "Right",
                    Self::Up => "Up",
                    Self::Down => "Down",
                    Self::Home => "Home",
                    Self::End => "End",
                    Self::PageUp => "PageUp",
                    Self::PageDown => "PageDown",
                    _ => "Insert",
                };
                f.write_str(name)
            }
        }
    }
}

/// Modifier keys held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modifiers {
    /// Shift.
    pub shift: bool,
    /// Control.
    pub ctrl: bool,
    /// Alt / Option.
    pub alt: bool,
    /// The logo key: Command on macOS, Windows key elsewhere.
    pub meta: bool,
}

impl Modifiers {
    /// None held.
    pub const NONE: Self = Self { shift: false, ctrl: false, alt: false, meta: false };
    /// Only Shift.
    pub const SHIFT: Self = Self { shift: true, ctrl: false, alt: false, meta: false };
    /// Only Control.
    pub const CTRL: Self = Self { shift: false, ctrl: true, alt: false, meta: false };

    /// The platform's command modifier: Command on macOS, Control elsewhere.
    #[inline]
    pub fn command(self) -> bool {
        if cfg!(target_os = "macos") { self.meta } else { self.ctrl }
    }
}

/// A pointer button.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointerButton {
    /// Left button, pen tip, touch.
    Primary,
    /// Right button, pen barrel button.
    Secondary,
    /// Middle button.
    Middle,
}

/// One input event, positions in logical pixels relative to the window's top-left corner.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The pointer moved.
    PointerMoved(Point),
    /// The pointer left the window.
    PointerLeft,
    /// A button went down or up.
    Button {
        /// Which button.
        button: PointerButton,
        /// Down (`true`) or up.
        pressed: bool,
        /// Where.
        pos: Point,
    },
    /// Scrolling as winit reports it: positive `y` moves the view towards the top (a wheel turned
    /// away from the user). In lines, or in logical pixels when `precise` (touchpads and
    /// high-resolution wheels; no line multiplier applied).
    Wheel {
        /// The distance.
        delta: Point,
        /// Pixel-precise (touchpad) rather than in lines.
        precise: bool,
    },
    /// Pen pressure and tilt where the platform reports them `[UI-06]`.
    Pen {
        /// Pressure, 0 to 1.
        pressure: f32,
        /// Tilt from vertical, radians, along X and Y.
        tilt: [f32; 2],
    },
    /// A key went down (`repeat` for auto-repeat) or up.
    Key {
        /// The key.
        key: Key,
        /// Down or up.
        pressed: bool,
        /// An auto-repeat of a held key.
        repeat: bool,
    },
    /// The modifiers changed.
    Modifiers(Modifiers),
    /// Typed text (after the keyboard layout and dead keys), one character.
    Text(char),
    /// The IME's composition string changed (empty when the composition ends).
    ImePreedit(String),
    /// The IME committed text.
    ImeCommit(String),
    /// The user asked to paste and this is the clipboard's text.
    Paste(String),
    /// The window gained or lost the keyboard focus.
    WindowFocus(bool),
    /// A finger on a touch screen `[UI-06]`, logical pixels. One finger acts as the primary
    /// pointer (tap = click, drag = drag, holding still for [`LONG_PRESS_SECONDS`] = secondary
    /// click); two fingers pan (as the wheel) and pinch ([`crate::ui::Ui::gesture`]).
    Touch {
        /// The finger (stable while it touches).
        id: u64,
        /// What it did.
        phase: TouchPhase,
        /// Where.
        pos: Point,
    },
}

/// What a finger did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TouchPhase {
    /// It touched the screen.
    Start,
    /// It moved.
    Move,
    /// It left the screen.
    End,
    /// The system took it away (no click).
    Cancel,
}

/// How long a still finger is held before it counts as a secondary click.
pub const LONG_PRESS_SECONDS: f64 = 0.5;

/// The events of one frame plus the clock, filled by the shell.
#[derive(Debug, Default)]
pub struct InputQueue {
    /// The events in arrival order.
    pub events: Vec<Event>,
    /// Seconds since an arbitrary start (monotonic; from `humanart_core::platform` clocks).
    pub time: f64,
}

impl InputQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an event.
    pub fn push(&mut self, event: Event) {
        self.events.push(event);
    }
}

/// A key with modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shortcut {
    /// The key.
    pub key: Key,
    /// The modifiers that must be held (and no others).
    pub mods: Modifiers,
}

impl Shortcut {
    /// A shortcut.
    pub const fn new(key: Key, mods: Modifiers) -> Self {
        Self { key, mods }
    }

    /// Writes `Ctrl+Shift+Z` style text (for menus, tooltips, the shortcut overlay).
    pub fn write(&self, out: &mut String) {
        for (held, name) in
            [(self.mods.ctrl, "Ctrl+"), (self.mods.alt, "Alt+"), (self.mods.shift, "Shift+"), (self.mods.meta, "Meta+")]
        {
            if held {
                out.push_str(name);
            }
        }
        self.key.write_name(out);
    }

    /// Parses `Ctrl+Shift+Z` style text.
    pub fn parse(text: &str, named: &[&'static str]) -> Option<Self> {
        let mut mods = Modifiers::NONE;
        let mut parts = text.split('+').peekable();
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                return Key::from_name(part, named).map(|key| Self { key, mods });
            }
            match part {
                "Ctrl" => mods.ctrl = true,
                "Alt" => mods.alt = true,
                "Shift" => mods.shift = true,
                "Meta" => mods.meta = true,
                _ => return None,
            }
        }
        None
    }
}

impl fmt::Display for Shortcut {
    /// `Ctrl+Shift+Z` style text, as [`Shortcut::write`] (no allocation).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (held, name) in
            [(self.mods.ctrl, "Ctrl+"), (self.mods.alt, "Alt+"), (self.mods.shift, "Shift+"), (self.mods.meta, "Meta+")]
        {
            if held {
                f.write_str(name)?;
            }
        }
        write!(f, "{}", self.key)
    }
}

/// The most bindings a keymap file may hold `[ROB-01]`.
pub const MAX_KEYMAP_BINDINGS: usize = 4096;
/// The longest shortcut text a keymap file may hold.
pub const MAX_SHORTCUT_TEXT: usize = 64;

/// A rebindable keymap `[UI-06]`: action names (stable ids such as `"edit.undo"`) to shortcuts.
/// An action may have several shortcuts; a shortcut triggers at most one action.
#[derive(Clone, Debug, Default)]
pub struct Keymap {
    bindings: Vec<(&'static str, Shortcut)>,
}

impl Keymap {
    /// An empty keymap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds `shortcut` to `action`, taking it away from any other action. Returns the action it
    /// was taken from.
    pub fn bind(&mut self, action: &'static str, shortcut: Shortcut) -> Option<&'static str> {
        let previous = self.bindings.iter().position(|b| b.1 == shortcut).map(|i| self.bindings.remove(i).0);
        self.bindings.push((action, shortcut));
        previous.filter(|p| *p != action)
    }

    /// Removes every shortcut of `action`.
    pub fn unbind(&mut self, action: &str) {
        self.bindings.retain(|b| b.0 != action);
    }

    /// The action bound to `shortcut`.
    pub fn action(&self, shortcut: Shortcut) -> Option<&'static str> {
        self.bindings.iter().find(|b| b.1 == shortcut).map(|b| b.0)
    }

    /// The shortcuts of `action`.
    pub fn shortcuts<'a>(&'a self, action: &'a str) -> impl Iterator<Item = Shortcut> + 'a {
        self.bindings.iter().filter(move |b| b.0 == action).map(|b| b.1)
    }

    /// Every binding, in binding order.
    pub fn bindings(&self) -> &[(&'static str, Shortcut)] {
        &self.bindings
    }

    /// The keymap as a keymap file: an object of action names to arrays of shortcut texts, in
    /// binding order (`{"edit.undo": ["Ctrl+Z"]}`).
    pub fn to_json(&self) -> Result<String, UiError> {
        let mut root = Object::new();
        let mut text = String::new();
        for (action, _) in &self.bindings {
            if root.position(action).is_some() {
                continue;
            }
            let mut list = Vec::new();
            for s in self.shortcuts(action) {
                text.clear();
                s.write(&mut text);
                list.push(Value::from(text.as_str()));
            }
            root.insert(*action, Value::Array(list));
        }
        json::to_string_pretty(&Value::from(root)).map_err(|e| UiError::Keymap(format!("{e}")))
    }

    /// Applies a keymap file (the user's rebindings): every action it names is rebound to exactly
    /// the shortcuts listed (an empty array unbinds it); actions it does not name keep theirs.
    /// `actions` lists the known action names and `named` the extra key names of
    /// [`Key::from_name`]. On any error (unknown action, bad shortcut, over a limit) the keymap is
    /// left unchanged.
    pub fn apply_json(&mut self, text: &str, actions: &[&'static str], named: &[&'static str]) -> Result<(), UiError> {
        let err = |what: String| UiError::Keymap(what);
        let value = json::parse_str(text).map_err(|e| err(format!("{e}")))?;
        let node = value.node();
        let mut next = self.clone();
        let mut total = 0_usize;
        for (name, list) in node.members().map_err(|e| err(format!("{e}")))? {
            let Some(action) = actions.iter().copied().find(|a| *a == name) else {
                return Err(err(format!("unknown action \"{name}\"")));
            };
            next.unbind(action);
            for item in list.elements().map_err(|e| err(format!("{e}")))? {
                let s = item.str().map_err(|e| err(format!("{e}")))?;
                if s.len() > MAX_SHORTCUT_TEXT {
                    return Err(err(format!("a shortcut of {action} is longer than {MAX_SHORTCUT_TEXT} bytes")));
                }
                let Some(shortcut) = Shortcut::parse(s, named) else {
                    return Err(err(format!("\"{s}\" of {action} is not a shortcut")));
                };
                total += 1;
                if total > MAX_KEYMAP_BINDINGS {
                    return Err(err(format!("more than {MAX_KEYMAP_BINDINGS} bindings")));
                }
                next.bind(action, shortcut);
            }
        }
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_06_shortcut_text_round_trips() {
        let named = ["Minus"];
        for (key, mods) in [
            (Key::Letter('Z'), Modifiers { ctrl: true, shift: true, ..Modifiers::NONE }),
            (Key::F(2), Modifiers::NONE),
            (Key::Digit('5'), Modifiers { alt: true, ..Modifiers::NONE }),
            (Key::Named("Minus"), Modifiers::CTRL),
            (Key::PageDown, Modifiers { meta: true, ..Modifiers::NONE }),
        ] {
            let s = Shortcut::new(key, mods);
            let mut text = String::new();
            s.write(&mut text);
            assert_eq!(Shortcut::parse(&text, &named), Some(s), "{text}");
        }
        assert_eq!(Shortcut::parse("Hyper+A", &named), None);
        assert_eq!(Shortcut::parse("Ctrl+", &named), None);
        assert_eq!(Shortcut::parse("F99", &named), None);
        assert_eq!(Shortcut::parse("Plus", &named), None);
    }

    #[test]
    fn ui_06_keymap_file_round_trips_and_rejects_bad_files() {
        let actions = ["edit.undo", "edit.redo", "view.frame"];
        let mut map = Keymap::new();
        map.bind("edit.undo", Shortcut::new(Key::Letter('Z'), Modifiers::CTRL));
        map.bind("edit.redo", Shortcut::new(Key::Letter('Y'), Modifiers::CTRL));
        map.bind(
            "edit.redo",
            Shortcut::new(Key::Letter('Z'), Modifiers { ctrl: true, shift: true, ..Modifiers::NONE }),
        );
        let text = map.to_json().unwrap();
        let mut back = Keymap::new();
        back.apply_json(&text, &actions, &[]).unwrap();
        assert_eq!(back.bindings(), map.bindings());
        // A user file rebinds one action and unbinds another; the rest stays.
        back.apply_json(r#"{"view.frame": ["F", "Named"], "edit.redo": []}"#, &actions, &["Named"]).unwrap();
        assert_eq!(back.action(Shortcut::new(Key::Letter('F'), Modifiers::NONE)), Some("view.frame"));
        assert_eq!(back.shortcuts("edit.redo").count(), 0);
        assert_eq!(back.shortcuts("edit.undo").count(), 1);
        // Bad files leave the keymap unchanged.
        let before = back.bindings().to_vec();
        for bad in [
            r#"{"nope": ["A"]}"#,
            r#"{"edit.undo": ["Hyper+Q"]}"#,
            r#"{"edit.undo": "A"}"#,
            "[",
            r#"{"edit.undo": [1]}"#,
        ] {
            assert!(back.apply_json(bad, &actions, &[]).is_err(), "{bad}");
            assert_eq!(back.bindings(), before.as_slice());
        }
        let long = format!(r#"{{"edit.undo": ["{}"]}}"#, "Ctrl+".repeat(20));
        assert!(back.apply_json(&long, &actions, &[]).is_err());
        // Display matches write.
        let s = Shortcut::new(Key::F(5), Modifiers { alt: true, shift: true, ..Modifiers::NONE });
        let mut w = String::new();
        s.write(&mut w);
        assert_eq!(format!("{s}"), w);
    }

    #[test]
    fn ui_06_keymap_rebinding_moves_a_shortcut() {
        let mut map = Keymap::new();
        let ctrl_z = Shortcut::new(Key::Letter('Z'), Modifiers::CTRL);
        assert_eq!(map.bind("edit.undo", ctrl_z), None);
        assert_eq!(map.action(ctrl_z), Some("edit.undo"));
        assert_eq!(map.bind("view.zoom", ctrl_z), Some("edit.undo"));
        assert_eq!(map.action(ctrl_z), Some("view.zoom"));
        assert_eq!(map.shortcuts("edit.undo").count(), 0);
        map.unbind("view.zoom");
        assert!(map.bindings().is_empty());
    }
}
