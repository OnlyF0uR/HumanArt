# Platform manual verification checklist (WP-31, `[AC-WP31-1]`)

What the automated tests in `crates/platform` cannot show, because it needs a person, the real
clipboard, a dialog on screen or a pen. Run it on each supported OS before a release and record the
result (date, OS version, machine, who) in the table at the end. Never record a box as passed that
was not tried.

Status of the implementation per OS (blocker B-003):

| Shim | Windows (x86_64 MSVC) | Linux (x86_64) | macOS (aarch64, x86_64) |
|---|---|---|---|
| `StdFileSystem`, `SystemClock`, `ThreadSpawner` | native, tested | std only; type-checked, never run | std only; type-checked, never run |
| `OsEntropy` | `BCryptGenRandom`, tested | `/dev/urandom`; never run | `getentropy`; never run |
| Clipboard text | `CF_UNICODETEXT`; read verified by hand, write only in an ignored test | in-process fallback only | in-process fallback only |
| File dialogs | `IFileOpenDialog` / `IFileSaveDialog` (COM); configuration tested, `Show` not run by an agent | `Unavailable` (in-app picker) | `Unavailable` (in-app picker) |
| DPI | winit (per-monitor v2) | winit | winit |
| Pen pressure / tilt | winit `Touch::force` + `GetPointerPenInfo` | winit `Touch::force` where the backend gives it, no tilt | winit `Touch::force` where the backend gives it, no tilt |

## 1. Automated part (every OS)

- [ ] `cargo test -p humanart-platform` passes.
- [ ] By hand, on a machine whose clipboard may be overwritten:
      `cargo test -p humanart-platform -- --ignored system_clipboard` passes (Windows; on Linux and
      macOS it reports the in-process fallback).
- [ ] By hand: `cargo test -p humanart-platform -- --ignored native_dialog_opens` shows Open, Open
      many, Save and Folder dialogs in turn (Windows); pick something in each, then repeat cancelling
      each. Every outcome printed is `Ok(Chosen([...]))` or `Ok(Cancelled)`.

## 2. File system (every OS)

- [ ] Save a document, kill the process (Task Manager / `kill -9`) during the next autosave: on
      restart the document opens with either the old or the new content, never a truncated one; a
      leftover `*.tmp~` file is ignored.
- [ ] Save into a folder on a USB stick and a network share; reopen.
- [ ] A path in a document pointing above its folder (`../x`) or (Windows) to `C:\...` is refused
      with an error message, not followed.
- [ ] File and folder names with accents, CJK and emoji appear correctly in the in-app browser and
      round-trip through save and open.
- [ ] Windows: a document inside a folder synchronised by OneDrive saves repeatedly without an
      "access denied" error (the rename retry).

## 3. Clipboard

- [ ] Copy a text field in the editor, paste it into a text editor of the OS: same text, line breaks
      intact (Windows: CRLF in Notepad).
- [ ] Copy multi-line text with emoji from the OS, paste into the editor: same text, LF line ends.
- [ ] Copy a picture in another program, paste into a text field: nothing is pasted, no error box.
- [ ] Keep the clipboard locked by another program (Windows: a clipboard manager stress tool):
      copying in the editor still works inside the editor; the log says why the system clipboard was
      bypassed.
- [ ] Shims disabled (preference "Use native platform services" off): copy and paste work inside the
      editor; pasting in another program does not get the editor's text.
- [ ] Linux / macOS (until native clipboards exist): copy and paste inside the editor work.

## 4. File dialogs

- [ ] Open: the filter list shows the editor's types first, then "All files"; only matching files
      are listed; double-click opens.
- [ ] Open many: Ctrl/Cmd-click several files; all are opened.
- [ ] Save: the proposed name is filled in; typing a name without extension appends the first
      filter's extension; choosing an existing file asks before replacing.
- [ ] Folder: only folders can be chosen.
- [ ] The dialog is modal to the editor window: the editor cannot be clicked while it is open, the
      editor keeps repainting (its event loop is not blocked), and the dialog closes cleanly when
      cancelled.
- [ ] The start folder is the last-used folder; a start folder that was deleted meanwhile opens the
      system's default instead of failing.
- [ ] Mixed-DPI monitors (Windows): the dialog opens at the right size on the monitor of the editor.
- [ ] Shims disabled, and on Linux / macOS: the in-app path picker appears instead, and opening and
      saving through it works.

## 5. DPI

- [ ] Start on a 100 % monitor, drag to a 150 % / 200 % (Retina) monitor: the UI rescales once,
      text stays crisp, no blur and no layout jump larger than one frame.
- [ ] Change the system scale while the editor runs: the UI follows.
- [ ] Viewport picking and gizmos land under the cursor at every scale.

## 6. Pen (a Wacom, Surface or similar pen)

- [ ] Brush strokes get thinner and lighter with light pressure, full with hard pressure.
- [ ] The brush cursor's tilt indicator follows the pen tilt (Windows); Linux / macOS show no tilt.
- [ ] The eraser end of the pen (or the eraser button) switches to the eraser tool where the tool
      supports it (Windows).
- [ ] The barrel button maps to the configured action (Windows).
- [ ] Fast strokes stay smooth (coalesced pointer frames are not dropped).
- [ ] Shims disabled: the pen still paints, with full pressure and no tilt, like a mouse.
- [ ] Mouse and touch input are unaffected by the pen shim.

Known gaps (from `crates/platform/src/pen.rs`): tilt is per pointer message, not per coalesced
history frame; a hovering pen gives no tilt or position through winit 0.30.

## Results

| Date | OS and version | Machine / pen | Section(s) | Result | By |
|---|---|---|---|---|---|
