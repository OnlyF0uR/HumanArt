//! The window system side `[UI-06]`: winit's events translated into the toolkit's input events (in
//! logical pixels), the toolkit's cursor shapes into winit's, and the window surface's
//! configuration for the editor's frame (the target the viewports are copied into).
//!
//! Keys: letters and digits come from the physical key (shortcuts stay where they are printed on a
//! QWERTY keyboard whatever the layout); the named keys (arrows, Enter, F1 ... F24) from the logical
//! key; typed text from the event's text, one [`Event::Text`] per character.

use humanart_ui::geom::Point;
use humanart_ui::input::{Event, Key, Modifiers, PointerButton};
use humanart_ui::ui::CursorIcon;
use winit::event::{ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, ModifiersState, NamedKey, PhysicalKey};

use crate::studio::FRAME_FORMAT;

/// The toolkit key of a winit key event, if the toolkit has one.
pub fn key_of(event: &KeyEvent) -> Option<Key> {
    if let PhysicalKey::Code(code) = event.physical_key
        && let Some(k) = code_key(code)
    {
        return Some(k);
    }
    match &event.logical_key {
        winit::keyboard::Key::Named(n) => named_key(*n),
        _ => None,
    }
}

fn code_key(code: KeyCode) -> Option<Key> {
    const LETTERS: [KeyCode; 26] = [
        KeyCode::KeyA,
        KeyCode::KeyB,
        KeyCode::KeyC,
        KeyCode::KeyD,
        KeyCode::KeyE,
        KeyCode::KeyF,
        KeyCode::KeyG,
        KeyCode::KeyH,
        KeyCode::KeyI,
        KeyCode::KeyJ,
        KeyCode::KeyK,
        KeyCode::KeyL,
        KeyCode::KeyM,
        KeyCode::KeyN,
        KeyCode::KeyO,
        KeyCode::KeyP,
        KeyCode::KeyQ,
        KeyCode::KeyR,
        KeyCode::KeyS,
        KeyCode::KeyT,
        KeyCode::KeyU,
        KeyCode::KeyV,
        KeyCode::KeyW,
        KeyCode::KeyX,
        KeyCode::KeyY,
        KeyCode::KeyZ,
    ];
    const DIGITS: [KeyCode; 10] = [
        KeyCode::Digit0,
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    if let Some(i) = LETTERS.iter().position(|c| *c == code) {
        return u8::try_from(i).ok().map(|i| Key::Letter(char::from(b'A' + i)));
    }
    if let Some(i) = DIGITS.iter().position(|c| *c == code) {
        return u8::try_from(i).ok().map(|i| Key::Digit(char::from(b'0' + i)));
    }
    None
}

fn named_key(n: NamedKey) -> Option<Key> {
    Some(match n {
        NamedKey::Tab => Key::Tab,
        NamedKey::Enter => Key::Enter,
        NamedKey::Escape => Key::Escape,
        NamedKey::Space => Key::Space,
        NamedKey::Backspace => Key::Backspace,
        NamedKey::Delete => Key::Delete,
        NamedKey::ArrowLeft => Key::Left,
        NamedKey::ArrowRight => Key::Right,
        NamedKey::ArrowUp => Key::Up,
        NamedKey::ArrowDown => Key::Down,
        NamedKey::Home => Key::Home,
        NamedKey::End => Key::End,
        NamedKey::PageUp => Key::PageUp,
        NamedKey::PageDown => Key::PageDown,
        NamedKey::Insert => Key::Insert,
        NamedKey::ContextMenu => Key::Named("ContextMenu"),
        NamedKey::F1 => Key::F(1),
        NamedKey::F2 => Key::F(2),
        NamedKey::F3 => Key::F(3),
        NamedKey::F4 => Key::F(4),
        NamedKey::F5 => Key::F(5),
        NamedKey::F6 => Key::F(6),
        NamedKey::F7 => Key::F(7),
        NamedKey::F8 => Key::F(8),
        NamedKey::F9 => Key::F(9),
        NamedKey::F10 => Key::F(10),
        NamedKey::F11 => Key::F(11),
        NamedKey::F12 => Key::F(12),
        _ => return None,
    })
}

/// The toolkit modifiers of winit's.
pub fn modifiers_of(m: ModifiersState) -> Modifiers {
    Modifiers { shift: m.shift_key(), ctrl: m.control_key(), alt: m.alt_key(), meta: m.super_key() }
}

/// Translates one window event into toolkit events (appended to `out`); `scale` is the window's
/// scale factor (physical pixels per logical pixel). Returns whether the event was input.
pub fn translate(event: &WindowEvent, scale: f64, pointer: &mut Point, out: &mut Vec<Event>) -> bool {
    #[allow(clippy::cast_possible_truncation)] // Window coordinates.
    let logical = |x: f64, y: f64| Point::new((x / scale) as f32, (y / scale) as f32);
    match event {
        WindowEvent::CursorMoved { position, .. } => {
            *pointer = logical(position.x, position.y);
            out.push(Event::PointerMoved(*pointer));
        }
        WindowEvent::CursorLeft { .. } => out.push(Event::PointerLeft),
        WindowEvent::MouseInput { state, button, .. } => {
            let button = match button {
                MouseButton::Left => PointerButton::Primary,
                MouseButton::Right => PointerButton::Secondary,
                MouseButton::Middle => PointerButton::Middle,
                _ => return true,
            };
            out.push(Event::Button { button, pressed: *state == ElementState::Pressed, pos: *pointer });
        }
        WindowEvent::MouseWheel { delta, .. } => {
            #[allow(clippy::cast_possible_truncation)] // Wheel deltas.
            let (delta, precise) = match delta {
                MouseScrollDelta::LineDelta(x, y) => (Point::new(*x, *y), false),
                MouseScrollDelta::PixelDelta(p) => (logical(p.x, p.y), true),
            };
            out.push(Event::Wheel { delta, precise });
        }
        WindowEvent::ModifiersChanged(m) => out.push(Event::Modifiers(modifiers_of(m.state()))),
        WindowEvent::KeyboardInput { event, .. } => {
            let pressed = event.state == ElementState::Pressed;
            if let Some(key) = key_of(event) {
                out.push(Event::Key { key, pressed, repeat: event.repeat });
            }
            if pressed && let Some(text) = &event.text {
                out.extend(text.chars().filter(|c| !c.is_control()).map(Event::Text));
            }
        }
        WindowEvent::Ime(Ime::Preedit(text, _)) => out.push(Event::ImePreedit(text.clone())),
        WindowEvent::Ime(Ime::Commit(text)) => out.push(Event::ImeCommit(text.clone())),
        WindowEvent::Focused(f) => out.push(Event::WindowFocus(*f)),
        _ => return false,
    }
    true
}

/// winit's cursor of a toolkit cursor.
pub fn cursor_of(c: CursorIcon) -> winit::window::CursorIcon {
    use winit::window::CursorIcon as W;
    match c {
        CursorIcon::Default => W::Default,
        CursorIcon::Pointer => W::Pointer,
        CursorIcon::Text => W::Text,
        CursorIcon::Grab => W::Grab,
        CursorIcon::Grabbing => W::Grabbing,
        CursorIcon::ResizeHorizontal => W::EwResize,
        CursorIcon::ResizeVertical => W::NsResize,
        CursorIcon::NotAllowed => W::NotAllowed,
    }
}

/// The surface configuration of the editor's window: the frame format as a render target (the
/// viewports are drawn inside the UI pass), or `None` when the surface does not offer it (the
/// caller then tries another backend).
pub fn surface_config(caps: &wgpu::SurfaceCapabilities, width: u32, height: u32) -> Option<wgpu::SurfaceConfiguration> {
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    if !caps.formats.contains(&FRAME_FORMAT) || !caps.usages.contains(usage) {
        return None;
    }
    Some(wgpu::SurfaceConfiguration {
        usage,
        format: FRAME_FORMAT,
        color_space: wgpu::SurfaceColorSpace::Auto,
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
        view_formats: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_06_physical_letters_and_digits_map_to_shortcut_keys() {
        assert_eq!(code_key(KeyCode::KeyZ), Some(Key::Letter('Z')));
        assert_eq!(code_key(KeyCode::KeyA), Some(Key::Letter('A')));
        assert_eq!(code_key(KeyCode::Digit0), Some(Key::Digit('0')));
        assert_eq!(code_key(KeyCode::Digit9), Some(Key::Digit('9')));
        assert_eq!(code_key(KeyCode::ShiftLeft), None);
        assert_eq!(named_key(NamedKey::F1), Some(Key::F(1)));
        assert_eq!(named_key(NamedKey::ArrowUp), Some(Key::Up));
        let m = modifiers_of(ModifiersState::CONTROL | ModifiersState::SHIFT);
        assert!(m.ctrl && m.shift && !m.alt);
    }

    #[test]
    fn surface_config_needs_the_frame_format_as_a_render_target() {
        let mut caps = wgpu::SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            usages: wgpu::TextureUsages::COPY_DST,
            ..Default::default()
        };
        assert!(surface_config(&caps, 800, 600).is_none());
        caps.formats.push(FRAME_FORMAT);
        assert!(surface_config(&caps, 800, 600).is_none(), "render access is needed");
        caps.usages |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        let c = surface_config(&caps, 0, 600).unwrap();
        assert_eq!((c.format, c.width, c.usage), (FRAME_FORMAT, 1, wgpu::TextureUsages::RENDER_ATTACHMENT));
    }
}
