//! [`FileDialogs`]: native open, save and folder dialogs `[APP-07]`, with [`ShimError::Disabled`] /
//! [`ShimError::Unavailable`] telling the application to show its own path picker instead
//! `[AC-WP31-1]`.
//!
//! Windows uses the common item dialog (`IFileOpenDialog` / `IFileSaveDialog`) through hand-written
//! COM vtables. Every dialog runs on its own short-lived thread, which initialises COM as a
//! single-threaded apartment and leaves the caller's thread untouched; [`FileDialogs::spawn`]
//! returns at once and the application polls [`PendingDialog::try_take`] once per frame, so the event
//! loop (which owns the dialog's owner window) keeps running.
//!
//! Linux and macOS have no native dialog here: [`FileDialogs::show`] reports
//! [`ShimError::Unavailable`] there (the reference application drives `zenity` / `kdialog` /
//! `osascript` helper processes; that is the way to add them, see `docs/platform-checklist.md`).

use std::sync::mpsc;

use crate::error::ShimError;

/// What the dialog asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialogKind {
    /// One existing file.
    Open,
    /// One or more existing files.
    OpenMany,
    /// A file name to save to; an existing file is confirmed before it is chosen.
    Save,
    /// One existing folder.
    Folder,
}

/// One entry of the dialog's file-type list: `Character files (*.anth)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileFilter {
    /// What the user reads.
    pub name: String,
    /// Extensions without the dot (`["anth", "json"]`); `"*"` alone matches every file.
    pub extensions: Vec<String>,
}

impl FileFilter {
    /// A filter called `name` for `extensions`.
    pub fn new(name: &str, extensions: &[&str]) -> Self {
        Self { name: name.to_owned(), extensions: extensions.iter().map(|e| (*e).to_owned()).collect() }
    }

    /// The pattern list the system wants: `*.anth;*.json`, or `*.*` for all files.
    pub fn pattern(&self) -> String {
        let mut out = String::new();
        for extension in &self.extensions {
            if !out.is_empty() {
                out.push(';');
            }
            out.push_str(if extension == "*" { "*.*" } else { "*." });
            if extension != "*" {
                out.push_str(extension);
            }
        }
        out
    }
}

/// A native window handle to own the dialog: on Windows the `HWND` (winit's
/// `Win32WindowHandle::hwnd`). The dialog is modal to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawWindow(pub isize);

/// Everything a dialog shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DialogRequest {
    /// Open, open many, save or folder.
    pub kind: DialogKind,
    /// The title bar text.
    pub title: String,
    /// The file-type list, first entry selected; ignored for [`DialogKind::Folder`]. When saving, the
    /// first extension of the first filter is appended to a name typed without one.
    pub filters: Vec<FileFilter>,
    /// The folder to start in (an absolute platform path); the system's choice when `None` or
    /// when it does not exist.
    pub start_dir: Option<String>,
    /// The proposed file name when saving.
    pub file_name: Option<String>,
    /// The window the dialog is modal to.
    pub owner: Option<RawWindow>,
}

impl DialogRequest {
    /// A request of `kind` with `title` and nothing else.
    pub fn new(kind: DialogKind, title: &str) -> Self {
        Self { kind, title: title.to_owned(), filters: Vec::new(), start_dir: None, file_name: None, owner: None }
    }

    /// Adds a file-type entry.
    pub fn with_filter(mut self, name: &str, extensions: &[&str]) -> Self {
        self.filters.push(FileFilter::new(name, extensions));
        self
    }

    /// Sets the folder to start in.
    pub fn in_dir(mut self, dir: &str) -> Self {
        self.start_dir = Some(dir.to_owned());
        self
    }

    /// Sets the proposed file name (save dialogs).
    pub fn named(mut self, file_name: &str) -> Self {
        self.file_name = Some(file_name.to_owned());
        self
    }

    /// Makes the dialog modal to `owner`.
    pub fn owned_by(mut self, owner: RawWindow) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Checks what the system would otherwise reject or misread: NUL characters, empty filters,
    /// extensions with separators or wildcards.
    pub fn validate(&self) -> Result<(), ShimError> {
        let texts = [Some(&self.title), self.start_dir.as_ref(), self.file_name.as_ref()];
        if texts.iter().flatten().any(|t| t.contains('\0')) {
            return Err(ShimError::InvalidRequest("text contains a NUL character"));
        }
        for filter in &self.filters {
            if filter.name.is_empty() || filter.name.contains('\0') {
                return Err(ShimError::InvalidRequest("a filter needs a name without NUL"));
            }
            if filter.extensions.is_empty() {
                return Err(ShimError::InvalidRequest("a filter needs at least one extension"));
            }
            let bad = |e: &String| e.is_empty() || (e != "*" && e.contains(['*', '?', ';', '.', '/', '\\', '\0', ' ']));
            if filter.extensions.iter().any(bad) {
                return Err(ShimError::InvalidRequest("an extension is empty or has a wildcard or separator"));
            }
        }
        Ok(())
    }

    /// The extension a save dialog appends to a bare name: the first of the first filter.
    pub fn default_extension(&self) -> Option<&str> {
        self.filters.first()?.extensions.first().map(String::as_str).filter(|e| *e != "*")
    }
}

/// How a dialog ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DialogOutcome {
    /// The user chose these absolute platform paths (one, except for [`DialogKind::OpenMany`]).
    Chosen(Vec<String>),
    /// The user closed the dialog without choosing.
    Cancelled,
}

/// A dialog running on its own thread.
#[derive(Debug)]
pub struct PendingDialog {
    state: Pending,
}

#[derive(Debug)]
enum Pending {
    Ready(Result<DialogOutcome, ShimError>),
    Waiting(mpsc::Receiver<Result<DialogOutcome, ShimError>>),
    Taken,
}

impl PendingDialog {
    fn ready(result: Result<DialogOutcome, ShimError>) -> Self {
        Self { state: Pending::Ready(result) }
    }

    /// The result once the dialog has closed, exactly once; `None` while it is open and after the
    /// result was taken. Never blocks: call it once per frame.
    pub fn try_take(&mut self) -> Option<Result<DialogOutcome, ShimError>> {
        match std::mem::replace(&mut self.state, Pending::Taken) {
            Pending::Ready(result) => Some(result),
            Pending::Waiting(receiver) => match receiver.try_recv() {
                Ok(result) => Some(result),
                Err(mpsc::TryRecvError::Empty) => {
                    self.state = Pending::Waiting(receiver);
                    None
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    Some(Err(ShimError::Thread("the dialog thread ended without a result".to_owned())))
                }
            },
            Pending::Taken => None,
        }
    }

    /// Blocks until the dialog closes. Never call this on the thread that owns the dialog's owner
    /// window: that thread must keep handling messages while the dialog is open.
    pub fn wait(mut self) -> Result<DialogOutcome, ShimError> {
        match std::mem::replace(&mut self.state, Pending::Taken) {
            Pending::Ready(result) => result,
            Pending::Waiting(receiver) => receiver
                .recv()
                .unwrap_or_else(|_| Err(ShimError::Thread("the dialog thread ended without a result".to_owned()))),
            Pending::Taken => Err(ShimError::Thread("the result was already taken".to_owned())),
        }
    }
}

/// The native dialogs, or their absence.
#[derive(Clone, Copy, Debug)]
pub struct FileDialogs {
    enabled: bool,
}

impl FileDialogs {
    /// The system's dialogs where this platform has a shim (Windows).
    #[must_use]
    pub const fn native() -> Self {
        Self { enabled: true }
    }

    /// No native dialogs: every request reports [`ShimError::Disabled`] and the application shows
    /// its own path picker.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }

    /// `true` when a request can open a native dialog.
    pub fn is_available(&self) -> bool {
        self.enabled && cfg!(windows)
    }

    /// Opens the dialog on its own thread and returns at once.
    pub fn spawn(&self, request: &DialogRequest) -> PendingDialog {
        if !self.enabled {
            return PendingDialog::ready(Err(ShimError::Disabled));
        }
        if !cfg!(windows) {
            return PendingDialog::ready(Err(ShimError::Unavailable));
        }
        if let Err(error) = request.validate() {
            return PendingDialog::ready(Err(error));
        }
        let (send, receive) = mpsc::channel();
        let request = request.clone();
        let started = std::thread::Builder::new().name("humanart-file-dialog".to_owned()).spawn(move || {
            // The receiver may be gone (the application stopped waiting); nothing to do then.
            let _ = send.send(native::show(&request));
        });
        match started {
            Ok(_) => PendingDialog { state: Pending::Waiting(receive) },
            Err(error) => PendingDialog::ready(Err(ShimError::Thread(error.to_string()))),
        }
    }

    /// Opens the dialog and blocks until it closes ([`PendingDialog::wait`]'s rule applies).
    pub fn show(&self, request: &DialogRequest) -> Result<DialogOutcome, ShimError> {
        self.spawn(request).wait()
    }
}

impl Default for FileDialogs {
    fn default() -> Self {
        Self::native()
    }
}

#[cfg(windows)]
mod native {
    use core::ffi::c_void;
    use core::ptr::null_mut;

    use super::{DialogKind, DialogOutcome, DialogRequest};
    use crate::error::ShimError;
    use crate::win::wide;

    type HResult = i32;

    #[repr(C)]
    pub(super) struct Guid(pub(super) u32, pub(super) u16, pub(super) u16, pub(super) [u8; 8]);

    const CLSID_FILE_OPEN_DIALOG: Guid =
        Guid(0xDC1C_5A9C, 0xE88A, 0x4DDE, [0xA5, 0xA1, 0x60, 0xF8, 0x2A, 0x20, 0xAE, 0xF7]);
    const CLSID_FILE_SAVE_DIALOG: Guid =
        Guid(0xC0B4_E2F3, 0xBA21, 0x4773, [0x8D, 0xBA, 0x33, 0x5E, 0xC9, 0x46, 0xEB, 0x8B]);
    pub(super) const IID_FILE_OPEN_DIALOG: Guid =
        Guid(0xD57C_7288, 0xD4AD, 0x4768, [0xBE, 0x02, 0x9D, 0x96, 0x95, 0x32, 0xD9, 0x60]);
    const IID_FILE_SAVE_DIALOG: Guid =
        Guid(0x84BC_CD23, 0x5FDE, 0x4CDB, [0xAE, 0xA4, 0xAF, 0x64, 0xB8, 0x3D, 0x78, 0xAB]);
    pub(super) const IID_SHELL_ITEM: Guid =
        Guid(0x4382_6D1E, 0xE718, 0x42EE, [0xBC, 0x55, 0xA1, 0xE2, 0x61, 0xC3, 0x7B, 0xFE]);

    const COINIT_APARTMENTTHREADED: u32 = 0x2;
    const COINIT_DISABLE_OLE1DDE: u32 = 0x4;
    const CLSCTX_INPROC_SERVER: u32 = 0x1;
    pub(super) const FOS_OVERWRITEPROMPT: u32 = 0x2;
    pub(super) const FOS_NOCHANGEDIR: u32 = 0x8;
    pub(super) const FOS_PICKFOLDERS: u32 = 0x20;
    pub(super) const FOS_FORCEFILESYSTEM: u32 = 0x40;
    pub(super) const FOS_ALLOWMULTISELECT: u32 = 0x200;
    pub(super) const FOS_PATHMUSTEXIST: u32 = 0x800;
    pub(super) const FOS_FILEMUSTEXIST: u32 = 0x1000;
    const SIGDN_FILESYSPATH: i32 = 0x8005_8000_u32 as i32;
    /// `HRESULT_FROM_WIN32(ERROR_CANCELLED)`: the user closed the dialog.
    const CANCELLED: HResult = 0x8007_04C7_u32 as HResult;
    /// The longest path the shim accepts back from the system, in UTF-16 units (the NTFS limit).
    const MAX_PATH_UNITS: usize = 32_768;

    #[link(name = "ole32")]
    unsafe extern "system" {
        fn CoInitializeEx(reserved: *mut c_void, flags: u32) -> HResult;
        safe fn CoUninitialize();
        fn CoCreateInstance(
            clsid: *const Guid,
            outer: *mut c_void,
            context: u32,
            iid: *const Guid,
            out: *mut *mut c_void,
        ) -> HResult;
        fn CoTaskMemFree(memory: *mut c_void);
    }

    #[link(name = "shell32")]
    unsafe extern "system" {
        fn SHCreateItemFromParsingName(
            path: *const u16,
            bind: *mut c_void,
            iid: *const Guid,
            out: *mut *mut c_void,
        ) -> HResult;
    }

    // The vtables list every slot in declaration order; slots this shim never calls are `usize`
    // (same size and alignment as a function pointer).

    #[repr(C)]
    pub(super) struct UnknownVtbl {
        query_interface: usize,
        add_ref: usize,
        release: unsafe extern "system" fn(*mut c_void) -> u32,
    }

    #[repr(C)]
    pub(super) struct FileDialogVtbl {
        unknown: UnknownVtbl,
        show: unsafe extern "system" fn(*mut c_void, isize) -> HResult,
        set_file_types: unsafe extern "system" fn(*mut c_void, u32, *const FilterSpec) -> HResult,
        set_file_type_index: unsafe extern "system" fn(*mut c_void, u32) -> HResult,
        pub(super) get_file_type_index: unsafe extern "system" fn(*mut c_void, *mut u32) -> HResult,
        advise: usize,
        unadvise: usize,
        set_options: unsafe extern "system" fn(*mut c_void, u32) -> HResult,
        pub(super) get_options: unsafe extern "system" fn(*mut c_void, *mut u32) -> HResult,
        set_default_folder: usize,
        set_folder: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HResult,
        pub(super) get_folder: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HResult,
        get_current_selection: usize,
        set_file_name: unsafe extern "system" fn(*mut c_void, *const u16) -> HResult,
        get_file_name: usize,
        set_title: unsafe extern "system" fn(*mut c_void, *const u16) -> HResult,
        set_ok_button_label: usize,
        set_file_name_label: usize,
        get_result: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HResult,
        add_place: usize,
        set_default_extension: unsafe extern "system" fn(*mut c_void, *const u16) -> HResult,
        close: usize,
        set_client_guid: usize,
        clear_client_data: usize,
        set_filter: usize,
    }

    #[repr(C)]
    pub(super) struct FileOpenDialogVtbl {
        dialog: FileDialogVtbl,
        get_results: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HResult,
        get_selected_items: usize,
    }

    #[repr(C)]
    pub(super) struct ShellItemVtbl {
        unknown: UnknownVtbl,
        bind_to_handler: usize,
        get_parent: usize,
        get_display_name: unsafe extern "system" fn(*mut c_void, i32, *mut *mut u16) -> HResult,
        get_attributes: usize,
        compare: usize,
    }

    #[repr(C)]
    pub(super) struct ShellItemArrayVtbl {
        unknown: UnknownVtbl,
        bind_to_handler: usize,
        get_property_store: usize,
        get_property_description_list: usize,
        get_attributes: usize,
        pub(super) get_count: unsafe extern "system" fn(*mut c_void, *mut u32) -> HResult,
        pub(super) get_item_at: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HResult,
        enum_items: usize,
    }

    /// `COMDLG_FILTERSPEC`.
    #[repr(C)]
    pub(super) struct FilterSpec {
        name: *const u16,
        spec: *const u16,
    }

    /// An owned COM interface pointer whose vtable is `V`; released on drop.
    pub(super) struct Com<V> {
        pub(super) raw: *mut *const V,
    }

    impl<V> Com<V> {
        /// Takes ownership of `raw` (one reference), or `None` for a null pointer.
        ///
        /// # Safety
        /// `raw` must be null or a live COM interface whose vtable starts with the layout `V`.
        pub(super) unsafe fn from_raw(raw: *mut c_void) -> Option<Self> {
            (!raw.is_null()).then_some(Self { raw: raw.cast() })
        }

        /// The vtable.
        pub(super) fn vtbl(&self) -> &V {
            // SAFETY: `raw` is a live interface (constructor contract): its first field points to a
            // vtable of layout `V` that lives as long as the object.
            unsafe { &**self.raw }
        }

        /// The `this` argument of every method.
        pub(super) fn this(&self) -> *mut c_void {
            self.raw.cast()
        }
    }

    impl<V> Drop for Com<V> {
        fn drop(&mut self) {
            // SAFETY: every vtable starts with IUnknown's three slots (`UnknownVtbl` is the
            // `repr(C)` prefix of each `V` used here), and this value owns one reference.
            unsafe {
                let unknown = &**self.raw.cast::<*const UnknownVtbl>();
                (unknown.release)(self.this());
            }
        }
    }

    fn check(call: &'static str, hr: HResult) -> Result<(), ShimError> {
        if hr < 0 { Err(ShimError::Os { call, code: i64::from(hr as u32) }) } else { Ok(()) }
    }

    /// COM initialised on this thread as a single-threaded apartment, until dropped.
    pub(super) struct Apartment;

    impl Apartment {
        pub(super) fn enter() -> Result<Self, ShimError> {
            // SAFETY: the reserved argument must be null; the flags are valid. A success (S_OK or
            // S_FALSE) is balanced by the CoUninitialize in Drop.
            let hr = unsafe { CoInitializeEx(null_mut(), COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) };
            check("CoInitializeEx", hr)?;
            Ok(Self)
        }
    }

    impl Drop for Apartment {
        fn drop(&mut self) {
            CoUninitialize();
        }
    }

    /// Creates an open (`save == false`) or save dialog object. Nothing is shown.
    pub(super) fn create_dialog(save: bool) -> Result<Com<FileDialogVtbl>, ShimError> {
        let (clsid, iid) = if save {
            (&CLSID_FILE_SAVE_DIALOG, &IID_FILE_SAVE_DIALOG)
        } else {
            (&CLSID_FILE_OPEN_DIALOG, &IID_FILE_OPEN_DIALOG)
        };
        let mut raw = null_mut();
        // SAFETY: the GUIDs are valid for the call and `raw` is a valid out pointer. The requested
        // interface (IFileOpenDialog / IFileSaveDialog) derives from IFileDialog, whose vtable
        // `FileDialogVtbl` describes.
        let hr = unsafe { CoCreateInstance(clsid, null_mut(), CLSCTX_INPROC_SERVER, iid, &raw mut raw) };
        check("CoCreateInstance", hr)?;
        // SAFETY: as above; a null result despite success is turned into an error.
        unsafe { Com::from_raw(raw) }.ok_or(ShimError::Os { call: "CoCreateInstance", code: 0 })
    }

    /// The shell item for an absolute file-system path.
    pub(super) fn shell_item(path: &str) -> Result<Com<ShellItemVtbl>, ShimError> {
        let path = wide(&path.replace('/', "\\")).ok_or(ShimError::InvalidRequest("path contains NUL"))?;
        let mut raw = null_mut();
        // SAFETY: `path` is NUL-terminated UTF-16 that outlives the call; the IID is IShellItem's,
        // which `ShellItemVtbl` describes; `raw` is a valid out pointer.
        let hr = unsafe { SHCreateItemFromParsingName(path.as_ptr(), null_mut(), &IID_SHELL_ITEM, &raw mut raw) };
        check("SHCreateItemFromParsingName", hr)?;
        // SAFETY: as above.
        unsafe { Com::from_raw(raw) }.ok_or(ShimError::Os { call: "SHCreateItemFromParsingName", code: 0 })
    }

    /// The file-system path of a shell item.
    pub(super) fn item_path(item: &Com<ShellItemVtbl>) -> Result<String, ShimError> {
        let mut text: *mut u16 = null_mut();
        // SAFETY: `item` is a live IShellItem; `text` is a valid out pointer that receives a
        // CoTaskMemAlloc'ed NUL-terminated string on success.
        let hr = unsafe { (item.vtbl().get_display_name)(item.this(), SIGDN_FILESYSPATH, &raw mut text) };
        check("IShellItem::GetDisplayName", hr)?;
        if text.is_null() {
            return Err(ShimError::Os { call: "IShellItem::GetDisplayName", code: 0 });
        }
        // SAFETY: `text` is a NUL-terminated string from the system: reading stops at the NUL or
        // at MAX_PATH_UNITS, whichever comes first, so it never reads past the allocation's end
        // (a path is never longer than MAX_PATH_UNITS). It is freed exactly once, after the copy.
        let units = unsafe {
            let mut len = 0;
            while len < MAX_PATH_UNITS && *text.add(len) != 0 {
                len += 1;
            }
            let copy = std::slice::from_raw_parts(text, len).to_vec();
            CoTaskMemFree(text.cast());
            copy
        };
        String::from_utf16(&units).map_err(|_| ShimError::NonUtf8Path)
    }

    /// The options for a dialog of `kind`, on top of what the dialog object starts with.
    pub(super) fn options_for(kind: DialogKind, base: u32) -> u32 {
        let common = base | FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR | FOS_PATHMUSTEXIST;
        match kind {
            DialogKind::Open => common | FOS_FILEMUSTEXIST,
            DialogKind::OpenMany => common | FOS_FILEMUSTEXIST | FOS_ALLOWMULTISELECT,
            DialogKind::Save => common | FOS_OVERWRITEPROMPT,
            DialogKind::Folder => common | FOS_PICKFOLDERS,
        }
    }

    /// The UTF-16 buffers a configured dialog points into; they must outlive it.
    pub(super) struct Texts {
        _strings: Vec<Vec<u16>>,
        _specs: Vec<FilterSpec>,
    }

    /// Applies everything in `request` except the owner to `dialog`.
    pub(super) fn configure(dialog: &Com<FileDialogVtbl>, request: &DialogRequest) -> Result<Texts, ShimError> {
        let invalid = || ShimError::InvalidRequest("text contains a NUL character");
        let vtbl = dialog.vtbl();
        let this = dialog.this();
        let mut strings = Vec::new();
        let mut specs = Vec::new();
        let mut options = 0;
        // SAFETY (whole block): `dialog` is a live IFileDialog; every pointer passed is either a
        // valid out pointer or NUL-terminated UTF-16 kept alive in `strings` (moved into the
        // returned `Texts`, which the caller holds until the dialog is released); `specs` likewise
        // holds the filter array, whose length is passed with it.
        unsafe {
            check("IFileDialog::GetOptions", (vtbl.get_options)(this, &raw mut options))?;
            check("IFileDialog::SetOptions", (vtbl.set_options)(this, options_for(request.kind, options)))?;
            let title = wide(&request.title).ok_or_else(invalid)?;
            check("IFileDialog::SetTitle", (vtbl.set_title)(this, title.as_ptr()))?;
            strings.push(title);
            if request.kind != DialogKind::Folder && !request.filters.is_empty() {
                for filter in &request.filters {
                    let name = wide(&format!("{} ({})", filter.name, filter.pattern())).ok_or_else(invalid)?;
                    let spec = wide(&filter.pattern()).ok_or_else(invalid)?;
                    specs.push(FilterSpec { name: name.as_ptr(), spec: spec.as_ptr() });
                    // Moving a Vec does not move its heap buffer: the pointers stay valid.
                    strings.push(name);
                    strings.push(spec);
                }
                let count = u32::try_from(specs.len()).map_err(|_| ShimError::InvalidRequest("too many filters"))?;
                check("IFileDialog::SetFileTypes", (vtbl.set_file_types)(this, count, specs.as_ptr()))?;
                check("IFileDialog::SetFileTypeIndex", (vtbl.set_file_type_index)(this, 1))?;
            }
            if request.kind == DialogKind::Save {
                if let Some(name) = &request.file_name {
                    let name = wide(name).ok_or_else(invalid)?;
                    check("IFileDialog::SetFileName", (vtbl.set_file_name)(this, name.as_ptr()))?;
                    strings.push(name);
                }
                if let Some(extension) = request.default_extension() {
                    let extension = wide(extension).ok_or_else(invalid)?;
                    check("IFileDialog::SetDefaultExtension", (vtbl.set_default_extension)(this, extension.as_ptr()))?;
                    strings.push(extension);
                }
            }
            // A start folder that does not exist is not worth failing the dialog for.
            if let Some(item) = request.start_dir.as_deref().and_then(|dir| shell_item(dir).ok()) {
                check("IFileDialog::SetFolder", (vtbl.set_folder)(this, item.this()))?;
            }
        }
        Ok(Texts { _strings: strings, _specs: specs })
    }

    /// Shows the dialog modally and collects the result. Runs on a dedicated thread.
    pub(super) fn show(request: &DialogRequest) -> Result<DialogOutcome, ShimError> {
        let _apartment = Apartment::enter()?;
        let dialog = create_dialog(request.kind == DialogKind::Save)?;
        let _texts = configure(&dialog, request)?;
        let owner = request.owner.map_or(0, |w| w.0);
        // SAFETY: `dialog` is live and configured; `owner` is 0 or a window handle the caller
        // vouches for (an invalid one makes Show fail, it does not corrupt memory).
        let hr = unsafe { (dialog.vtbl().show)(dialog.this(), owner) };
        if hr == CANCELLED {
            return Ok(DialogOutcome::Cancelled);
        }
        check("IFileDialog::Show", hr)?;
        let mut raw = null_mut();
        if request.kind == DialogKind::OpenMany {
            // SAFETY: an OpenMany dialog was created as IFileOpenDialog (`create_dialog(false)`),
            // whose vtable extends IFileDialog's with GetResults; `raw` is a valid out pointer.
            unsafe {
                let open = &*dialog.raw.cast::<*const FileOpenDialogVtbl>().read();
                check("IFileOpenDialog::GetResults", (open.get_results)(dialog.this(), &raw mut raw))?;
            }
            // SAFETY: GetResults returns an IShellItemArray.
            let array: Com<ShellItemArrayVtbl> =
                unsafe { Com::from_raw(raw) }.ok_or(ShimError::Os { call: "IFileOpenDialog::GetResults", code: 0 })?;
            array_paths(&array).map(DialogOutcome::Chosen)
        } else {
            // SAFETY: `dialog` is live; `raw` is a valid out pointer that receives an IShellItem.
            let hr = unsafe { (dialog.vtbl().get_result)(dialog.this(), &raw mut raw) };
            check("IFileDialog::GetResult", hr)?;
            // SAFETY: as above.
            let item: Com<ShellItemVtbl> =
                unsafe { Com::from_raw(raw) }.ok_or(ShimError::Os { call: "IFileDialog::GetResult", code: 0 })?;
            Ok(DialogOutcome::Chosen(vec![item_path(&item)?]))
        }
    }

    /// The paths of every item in a shell item array.
    pub(super) fn array_paths(array: &Com<ShellItemArrayVtbl>) -> Result<Vec<String>, ShimError> {
        let mut count = 0;
        // SAFETY: `array` is a live IShellItemArray; `count` is a valid out pointer.
        check("IShellItemArray::GetCount", unsafe { (array.vtbl().get_count)(array.this(), &raw mut count) })?;
        let mut paths = Vec::new();
        for index in 0..count {
            let mut raw = null_mut();
            // SAFETY: `index` < count; `raw` is a valid out pointer that receives an IShellItem.
            let hr = unsafe { (array.vtbl().get_item_at)(array.this(), index, &raw mut raw) };
            check("IShellItemArray::GetItemAt", hr)?;
            // SAFETY: as above.
            let item: Com<ShellItemVtbl> =
                unsafe { Com::from_raw(raw) }.ok_or(ShimError::Os { call: "IShellItemArray::GetItemAt", code: 0 })?;
            paths.push(item_path(&item)?);
        }
        Ok(paths)
    }
}

#[cfg(not(windows))]
mod native {
    use super::{DialogOutcome, DialogRequest};
    use crate::error::ShimError;

    pub(super) fn show(_request: &DialogRequest) -> Result<DialogOutcome, ShimError> {
        Err(ShimError::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn character_request(kind: DialogKind) -> DialogRequest {
        DialogRequest::new(kind, "Open character")
            .with_filter("Character", &["anth"])
            .with_filter("Any file", &["*"])
            .named("untitled")
    }

    #[test]
    fn disabled_dialogs_report_disabled_so_the_app_shows_its_own_picker_ac_wp31_1() {
        let dialogs = FileDialogs::disabled();
        assert!(!dialogs.is_available());
        assert_eq!(dialogs.show(&character_request(DialogKind::Open)), Err(ShimError::Disabled));
        let mut pending = dialogs.spawn(&character_request(DialogKind::Folder));
        assert_eq!(pending.try_take(), Some(Err(ShimError::Disabled)));
        assert_eq!(pending.try_take(), None);
    }

    #[test]
    fn requests_are_validated_before_anything_is_shown() {
        assert!(character_request(DialogKind::Save).validate().is_ok());
        let bad = [
            DialogRequest::new(DialogKind::Open, "a\0b"),
            DialogRequest::new(DialogKind::Open, "t").with_filter("", &["x"]),
            DialogRequest::new(DialogKind::Open, "t").with_filter("X", &[]),
            DialogRequest::new(DialogKind::Open, "t").with_filter("X", &["*.x"]),
            DialogRequest::new(DialogKind::Open, "t").with_filter("X", &["a;b"]),
            DialogRequest::new(DialogKind::Open, "t").with_filter("X", &[""]),
            DialogRequest::new(DialogKind::Save, "t").named("a\0"),
        ];
        for request in &bad {
            assert!(matches!(request.validate(), Err(ShimError::InvalidRequest(_))), "{request:?}");
            // An invalid request never reaches the system.
            assert!(matches!(
                FileDialogs::native().show(request),
                Err(ShimError::InvalidRequest(_) | ShimError::Unavailable)
            ));
        }
    }

    #[test]
    fn filters_become_system_patterns() {
        assert_eq!(FileFilter::new("Images", &["png", "exr"]).pattern(), "*.png;*.exr");
        assert_eq!(FileFilter::new("All", &["*"]).pattern(), "*.*");
        assert_eq!(character_request(DialogKind::Save).default_extension(), Some("anth"));
        assert_eq!(DialogRequest::new(DialogKind::Save, "t").with_filter("All", &["*"]).default_extension(), None);
        assert_eq!(DialogRequest::new(DialogKind::Save, "t").default_extension(), None);
    }

    /// The COM vtables at their boundary: a dialog object is created and configured, its state is
    /// read back through other slots, and shell items and arrays round-trip a real path. Nothing is
    /// shown (no window exists until `Show`).
    #[cfg(windows)]
    #[test]
    fn com_vtables_configure_a_dialog_without_showing_it() {
        use native::*;
        const IID_SHELL_ITEM_ARRAY: Guid =
            Guid(0xB63E_A76D, 0x1F85, 0x456F, [0xA1, 0x9C, 0x48, 0x15, 0x9E, 0xFA, 0x85, 0x8B]);
        #[link(name = "shell32")]
        unsafe extern "system" {
            fn SHCreateShellItemArrayFromShellItem(
                item: *mut core::ffi::c_void,
                iid: *const Guid,
                out: *mut *mut core::ffi::c_void,
            ) -> i32;
        }
        let dir = crate::fs::tests::TempDir::new("dialog");
        let dir_text = dir.path().to_str().unwrap().to_owned();
        let worker = std::thread::spawn(move || {
            let _apartment = Apartment::enter().unwrap();
            for kind in [DialogKind::Open, DialogKind::OpenMany, DialogKind::Save, DialogKind::Folder] {
                let dialog = create_dialog(kind == DialogKind::Save).unwrap();
                let request = character_request(kind).in_dir(&dir_text);
                let _texts = configure(&dialog, &request).unwrap();
                let mut options = 0;
                // SAFETY: live dialog, valid out pointer.
                unsafe { (dialog.vtbl().get_options)(dialog.this(), &raw mut options) };
                assert_eq!(options & options_for(kind, 0), options_for(kind, 0), "{kind:?}");
                if kind != DialogKind::Folder {
                    let mut index = 0;
                    // SAFETY: live dialog, valid out pointer.
                    unsafe { (dialog.vtbl().get_file_type_index)(dialog.this(), &raw mut index) };
                    assert_eq!(index, 1);
                }
                let mut folder = core::ptr::null_mut();
                // SAFETY: live dialog, valid out pointer receiving an IShellItem.
                let hr = unsafe { (dialog.vtbl().get_folder)(dialog.this(), &raw mut folder) };
                assert!(hr >= 0);
                // SAFETY: as above.
                let folder: Com<ShellItemVtbl> = unsafe { Com::from_raw(folder) }.unwrap();
                assert_eq!(item_path(&folder).unwrap().to_lowercase(), dir_text.to_lowercase());
            }
            let item = shell_item(&dir_text).unwrap();
            let mut raw = core::ptr::null_mut();
            // SAFETY: live item, the IID matches the out pointer's interface.
            let hr = unsafe { SHCreateShellItemArrayFromShellItem(item.this(), &IID_SHELL_ITEM_ARRAY, &raw mut raw) };
            assert!(hr >= 0);
            // SAFETY: as above.
            let array: Com<ShellItemArrayVtbl> = unsafe { Com::from_raw(raw) }.unwrap();
            let paths = array_paths(&array).unwrap();
            assert_eq!(paths.len(), 1);
            assert_eq!(paths[0].to_lowercase(), dir_text.to_lowercase());
            assert!(shell_item("Z:/surely/not/here/humanart").is_err());
        });
        worker.join().unwrap();
    }

    /// Opens a real dialog on the owner's screen. Run by hand only:
    /// `cargo test -p humanart-platform -- --ignored native_dialog_opens`.
    #[test]
    #[ignore = "opens a dialog on the user's screen"]
    fn native_dialog_opens() {
        let dialogs = FileDialogs::native();
        for kind in [DialogKind::Open, DialogKind::OpenMany, DialogKind::Save, DialogKind::Folder] {
            let outcome = dialogs.show(&character_request(kind));
            eprintln!("{kind:?}: {outcome:?}");
            assert!(outcome.is_ok());
        }
    }
}
