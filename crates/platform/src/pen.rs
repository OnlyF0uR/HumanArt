//! [`PenInput`]: pen pressure and tilt `[UI-06]`, with mouse semantics (pressure 1.0, no tilt) as
//! the fallback `[AC-WP31-1]`.
//!
//! Windows: winit already handles `WM_POINTER` messages and turns a pen into
//! `WindowEvent::Touch { id, force, .. }`, with `id` the system pointer id and `force` the pen's
//! pressure, but it drops tilt, rotation and the barrel and eraser buttons. While the application
//! handles that event, the message is still the thread's current pointer message, so
//! [`PenInput::touch_sample`] asks `GetPointerPenInfo(id)` for the rest. No window procedure is
//! replaced or subclassed. Two limits follow from that:
//!
//! - winit expands a coalesced `WM_POINTERUPDATE` into one `Touch` event per history frame;
//!   `GetPointerPenInfo` describes the newest frame, so tilt is per message, not per history frame
//!   (pressure from winit's `force` stays per frame).
//! - A pen hovering without contact produces no `Touch` event in winit 0.30, so there is no hover
//!   tilt or hover position from the pen (the system still moves the mouse cursor).
//!
//! Linux and macOS: winit 0.30 reports pen pressure as `Touch::force` on some backends and tilt on
//! none; [`PenInput::touch_sample`] uses `force` there and reports no tilt.

/// What produced a pointer sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerKind {
    /// A mouse, or a pen while the shim is disabled.
    Mouse,
    /// A pen tip.
    Pen,
    /// A pen's eraser end (or its eraser button held).
    Eraser,
    /// A finger, or a pen the system could not describe.
    Touch,
}

/// One pointer sample for brushes and sculpt tools.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenSample {
    /// What produced it.
    pub kind: PointerKind,
    /// Normalised pressure, 0..=1 (1.0 when the device reports none).
    pub pressure: f32,
    /// Tilt towards +X (right) in degrees, -90..=90; 0 when unknown.
    pub tilt_x_deg: f32,
    /// Tilt towards +Y (towards the user) in degrees, -90..=90; 0 when unknown.
    pub tilt_y_deg: f32,
    /// Clockwise rotation about the pen's axis in degrees, 0..360; 0 when unknown.
    pub twist_deg: f32,
    /// The barrel button is held.
    pub barrel_button: bool,
    /// `pressure` came from the device.
    pub has_pressure: bool,
    /// The tilt fields came from the device.
    pub has_tilt: bool,
}

impl PenSample {
    /// A mouse: full pressure, upright, no buttons.
    pub const MOUSE: Self = Self {
        kind: PointerKind::Mouse,
        pressure: 1.0,
        tilt_x_deg: 0.0,
        tilt_y_deg: 0.0,
        twist_deg: 0.0,
        barrel_button: false,
        has_pressure: false,
        has_tilt: false,
    };

    /// A touch whose only extra information is winit's normalised `force`.
    pub fn from_force(force: Option<f64>) -> Self {
        let pressure = force.filter(|f| f.is_finite()).map(|f| f.clamp(0.0, 1.0) as f32);
        Self {
            kind: PointerKind::Touch,
            pressure: pressure.unwrap_or(1.0),
            has_pressure: pressure.is_some(),
            ..Self::MOUSE
        }
    }
}

/// The raw fields of a Windows `POINTER_PEN_INFO` that matter here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawPenInfo {
    /// `PEN_FLAG_*`.
    pub flags: u32,
    /// `PEN_MASK_*`: which of the following are valid.
    pub mask: u32,
    /// 0..=1024.
    pub pressure: u32,
    /// Degrees, 0..=359.
    pub rotation: u32,
    /// Degrees, -90..=90.
    pub tilt_x: i32,
    /// Degrees, -90..=90.
    pub tilt_y: i32,
}

impl RawPenInfo {
    const FLAG_BARREL: u32 = 0x1;
    const FLAG_INVERTED: u32 = 0x2;
    const FLAG_ERASER: u32 = 0x4;
    const MASK_PRESSURE: u32 = 0x1;
    const MASK_ROTATION: u32 = 0x2;
    const MASK_TILT_X: u32 = 0x4;
    const MASK_TILT_Y: u32 = 0x8;

    /// The sample these fields describe; out-of-range values (a faulty driver) are clamped.
    pub fn to_sample(self) -> PenSample {
        let has_pressure = self.mask & Self::MASK_PRESSURE != 0;
        let has_tilt = self.mask & (Self::MASK_TILT_X | Self::MASK_TILT_Y) != 0;
        let eraser = self.flags & (Self::FLAG_INVERTED | Self::FLAG_ERASER) != 0;
        let tilt = |mask: u32, value: i32| if self.mask & mask != 0 { value.clamp(-90, 90) as f32 } else { 0.0 };
        PenSample {
            kind: if eraser { PointerKind::Eraser } else { PointerKind::Pen },
            pressure: if has_pressure { self.pressure.min(1024) as f32 / 1024.0 } else { 1.0 },
            tilt_x_deg: tilt(Self::MASK_TILT_X, self.tilt_x),
            tilt_y_deg: tilt(Self::MASK_TILT_Y, self.tilt_y),
            twist_deg: if self.mask & Self::MASK_ROTATION != 0 { (self.rotation % 360) as f32 } else { 0.0 },
            barrel_button: self.flags & Self::FLAG_BARREL != 0,
            has_pressure,
            has_tilt,
        }
    }
}

/// Pen input, or mouse semantics when disabled.
#[derive(Clone, Copy, Debug)]
pub struct PenInput {
    enabled: bool,
}

impl PenInput {
    /// Pen details where the platform has a shim (Windows), winit's `force` elsewhere.
    #[must_use]
    pub const fn native() -> Self {
        Self { enabled: true }
    }

    /// Every pointer behaves like a mouse: pressure 1.0, no tilt.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }

    /// `true` when tilt and the pen's buttons can be read.
    pub fn is_available(&self) -> bool {
        self.enabled && cfg!(windows)
    }

    /// The sample for winit's `WindowEvent::Touch { id, force, .. }`. Call it while handling that
    /// event (the system answers for the pointer message being handled), with `force` as
    /// `touch.force.map(|f| f.normalized())`.
    pub fn touch_sample(&self, touch_id: u64, force: Option<f64>) -> PenSample {
        if !self.enabled {
            return PenSample::MOUSE;
        }
        match u32::try_from(touch_id).ok().and_then(native::pen_info) {
            Some(info) => {
                let mut sample = info.to_sample();
                // winit's force is per history frame; the system's answer only for the newest.
                let winit = PenSample::from_force(force);
                if winit.has_pressure {
                    sample.pressure = winit.pressure;
                    sample.has_pressure = true;
                }
                sample
            }
            None => PenSample::from_force(force),
        }
    }

    /// The sample for a mouse event.
    pub fn mouse_sample(&self) -> PenSample {
        PenSample::MOUSE
    }
}

impl Default for PenInput {
    fn default() -> Self {
        Self::native()
    }
}

#[cfg(windows)]
mod native {
    use core::ffi::c_void;

    use super::RawPenInfo;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Point {
        x: i32,
        y: i32,
    }

    /// `POINTER_INFO`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct PointerInfo {
        pointer_type: u32,
        pointer_id: u32,
        frame_id: u32,
        pointer_flags: u32,
        source_device: *mut c_void,
        hwnd_target: *mut c_void,
        pixel_location: Point,
        himetric_location: Point,
        pixel_location_raw: Point,
        himetric_location_raw: Point,
        time: u32,
        history_count: u32,
        input_data: i32,
        key_states: u32,
        performance_count: u64,
        button_change_type: i32,
    }

    /// `POINTER_PEN_INFO`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct PointerPenInfo {
        pointer_info: PointerInfo,
        pen_flags: u32,
        pen_mask: u32,
        pressure: u32,
        rotation: u32,
        tilt_x: i32,
        tilt_y: i32,
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetPointerPenInfo(pointer_id: u32, info: *mut PointerPenInfo) -> i32;
    }

    /// The pen details of pointer `id` for the current pointer message, `None` when `id` is not a
    /// pen (or no longer known).
    pub(super) fn pen_info(id: u32) -> Option<RawPenInfo> {
        let mut info = core::mem::MaybeUninit::<PointerPenInfo>::zeroed();
        // SAFETY: `info` is a writable, correctly sized and aligned POINTER_PEN_INFO (the layout
        // is checked by a test); the call writes only inside it and reports failure with 0.
        let ok = unsafe { GetPointerPenInfo(id, info.as_mut_ptr()) };
        if ok == 0 {
            return None;
        }
        // SAFETY: zero-initialised and then filled by the system; every field is plain data for
        // which any bit pattern is valid.
        let info = unsafe { info.assume_init() };
        Some(RawPenInfo {
            flags: info.pen_flags,
            mask: info.pen_mask,
            pressure: info.pressure,
            rotation: info.rotation,
            tilt_x: info.tilt_x,
            tilt_y: info.tilt_y,
        })
    }
}

#[cfg(not(windows))]
mod native {
    use super::RawPenInfo;

    pub(super) fn pen_info(_id: u32) -> Option<RawPenInfo> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_pen_input_behaves_like_a_mouse_ac_wp31_1() {
        let pen = PenInput::disabled();
        assert!(!pen.is_available());
        let sample = pen.touch_sample(1, Some(0.3));
        assert_eq!(sample, PenSample::MOUSE);
        assert_eq!(sample.pressure, 1.0);
        assert_eq!(pen.mouse_sample(), PenSample::MOUSE);
    }

    #[test]
    fn touches_without_pen_details_use_the_force_or_full_pressure() {
        let pen = PenInput::native();
        // Pointer ids the system never hands out: no pen info, so winit's force decides.
        let sample = pen.touch_sample(u64::from(u32::MAX) + 5, Some(0.25));
        assert_eq!(
            (sample.kind, sample.pressure, sample.has_pressure, sample.has_tilt),
            (PointerKind::Touch, 0.25, true, false)
        );
        let sample = pen.touch_sample(0xFFFF_FFF0, None);
        assert_eq!((sample.kind, sample.pressure, sample.has_pressure), (PointerKind::Touch, 1.0, false));
        assert_eq!(PenSample::from_force(Some(7.0)).pressure, 1.0);
        assert_eq!(PenSample::from_force(Some(-1.0)).pressure, 0.0);
        assert_eq!(PenSample::from_force(Some(f64::NAN)).pressure, 1.0);
    }

    #[test]
    fn raw_pen_info_becomes_a_sample() {
        let tip =
            RawPenInfo { flags: 0x1, mask: 0xF, pressure: 512, rotation: 370, tilt_x: 30, tilt_y: -120 }.to_sample();
        assert_eq!(tip.kind, PointerKind::Pen);
        assert_eq!(tip.pressure, 0.5);
        assert_eq!((tip.tilt_x_deg, tip.tilt_y_deg, tip.twist_deg), (30.0, -90.0, 10.0));
        assert!(tip.barrel_button && tip.has_pressure && tip.has_tilt);
        let eraser = RawPenInfo { flags: 0x4, mask: 0x1, pressure: 5000, ..RawPenInfo::default() }.to_sample();
        assert_eq!(
            (eraser.kind, eraser.pressure, eraser.has_tilt, eraser.tilt_x_deg),
            (PointerKind::Eraser, 1.0, false, 0.0)
        );
        let inverted = RawPenInfo { flags: 0x2, mask: 0, tilt_x: 40, ..RawPenInfo::default() }.to_sample();
        assert_eq!(
            (inverted.kind, inverted.pressure, inverted.has_pressure, inverted.tilt_x_deg),
            (PointerKind::Eraser, 1.0, false, 0.0)
        );
    }

    /// The FFI boundary: the struct has the size Windows expects (a wrong size makes the call fail
    /// or write past it) and an unknown pointer id is refused, not misread.
    #[cfg(windows)]
    #[test]
    fn pointer_pen_info_matches_the_windows_layout() {
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::size_of::<native::PointerInfo>(), 96);
            assert_eq!(std::mem::size_of::<native::PointerPenInfo>(), 120);
        }
        assert!(native::pen_info(0xFFFF_FFF0).is_none());
    }
}
