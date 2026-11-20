# UI design tokens

`[UI-08]` The design-token system of `humanart-ui`: every colour, size, text style, shadow and timing a
widget uses comes from the active `Theme` (`crates/ui/src/theme.rs`); widget code never hard-codes
one. A theme is plain data (`Theme`, `Copy`), so switching themes, cross-fading (`Theme::mix`) and
hot-reloading a theme file are value assignments.

The values follow the reference application's tokens (`docs/reference-conventions.md` section 16):
its Black scheme is the dark theme, its White scheme the light theme. HumanArt adds the tokens an
editor needs (focus ring, warning, viewport surround, slider and switch sizes) and a high-contrast
theme `[UI-09]`. The type scale is about one pixel smaller than the reference's chat-reading sizes,
because the editor is dense with property rows.

## Built-in themes

| Theme | Constructor | Name in files |
|---|---|---|
| Dark (default) | `Theme::dark()` | `"dark"` |
| Light | `Theme::light()` | `"light"` |
| High contrast | `Theme::high_contrast()` | `"high-contrast"` |

Modifiers on any theme:

- **Colour-neutral mode** (`Theme::colour_neutral()`): every palette colour becomes the grey of the
  same relative luminance (alpha kept) and the viewport surround becomes the 18 % linear mid grey
  (`Color::MID_GREY`, sRGB `#767676`), so nothing near the viewport tints the perception of skin
  colour. The surround of every normal theme is already a hue-free grey. Idempotent.
- **Reduced motion** (`Theme::with_reduced_motion(true)`, `[UI-09]`): eases jump to their target and
  the caret does not blink.
- **UI scale** (75 to 200 %, `[UI-09]`) is not a token: the painter multiplies logical pixels by
  display scale times UI scale, and pixel-snaps edges.

Contrast guarantees, checked by `theme::tests::ui_08_text_contrast_on_surfaces`: primary text is at
least 7:1 on `bg`, `panel`, `surface` and `hover` in every theme, muted text at least 4.5:1; in the
high-contrast theme muted text and the focus ring are at least 7:1.

## Colours

Colours are written as sRGB hex (`@a` is a straight alpha) and stored as linear Rec.709
(`Color::hex` decodes at compile time). Blending happens in display space, as in the reference.

| Token | Role | Dark | Light | High contrast |
|---|---|---|---|---|
| `bg` | The main canvas behind panels' content. | `#121212` | `#FFFFFF` | `#000000` |
| `panel` | Window chrome, docked panels; also the clear colour. | `#080808` | `#F4F4F5` | `#000000` |
| `surface` | Raised surfaces: cards, menus, popovers, text fields. | `#1A1A1A` | `#FFFFFF` | `#0A0A0A` |
| `hover` | A hovered element. | `#1F1F1F` | `#ECECEE` | `#262626` |
| `active` | A selected or pressed element; slider and switch tracks. | `#2A2A2A` | `#E4E4E7` | `#3A3A3A` |
| `border` | Hairline dividers. | `#222222` | `#E6E6E9` | `#8A8A8A` |
| `border_strong` | Borders of controls and floating surfaces. | `#303030` | `#D6D6DB` | `#D0D0D0` |
| `border_focus` | The border of a focused input. | `#4C4C4C` | `#A1A1AA` | `#FFFFFF` |
| `text` | Primary text. | `#EDEDED` | `#18181B` | `#FFFFFF` |
| `text_muted` | Secondary text. | `#A3A3A3` | `#55555E` | `#E0E0E0` |
| `text_faint` | Hints, placeholders, disabled text. | `#6B6B6B` | `#9B9BA4` | `#B8B8B8` |
| `accent` | The single accent colour (primary buttons, checked switches, slider fill). | `#EDEDED` | `#18181B` | `#FFFFFF` |
| `on_accent` | Text and icons on the accent colour. | `#0A0A0A` | `#FAFAFA` | `#000000` |
| `selection` | Text selection and selected rows. | `#FFFFFF` @0.16 | `#2563EB` @0.16 | `#FFD400` @0.35 |
| `focus_ring` | The keyboard focus ring `[UI-09]`. | `#7DB3FF` | `#2563EB` | `#FFD400` |
| `danger` | Errors and destructive actions. | `#F26D6D` | `#DC2626` | `#FF8080` |
| `positive` | Success. | `#6CCB8B` | `#15803D` | `#7CF29A` |
| `warning` | Warnings. | `#E8B04A` | `#B45309` | `#FFD400` |
| `link` | Links. | `#7DB3FF` | `#2563EB` | `#9CCBFF` |
| `shadow` | Drop shadows under floating surfaces. | `#000000` @0.6 | `#0F0F14` @0.09 | `#000000` @0.0 |
| `scrim` | The modal dimmer. | `#000000` @0.6 | `#0C0C0F` @0.55 | `#000000` @0.8 |
| `on_scrim` | Text on the scrim. | `#F5F5F5` | `#FFFFFF` | `#FFFFFF` |
| `viewport_surround` | The area around a viewport's image; always a hue-free grey. | `#2B2B2B` | `#BDBDBD` | `#3A3A3A` |

## Sizes

Logical pixels; the same in every theme except where the high-contrast theme widens them
(`focus_ring_width` 3, `caret_width` 2, `scrollbar_thin` 5, `scrollbar_thick` 8).

| Token | Use | Value |
|---|---|---|
| `radius_sm` | Small radius: list rows, buttons, text fields. | 7.0 |
| `radius` | Default radius: menus, cards, tooltips. | 10.0 |
| `radius_lg` | Large radius: modals, large cards. | 16.0 |
| `space_1` | Spacing step 1 (hairline gaps). | 2.0 |
| `space_2` | Spacing step 2. | 4.0 |
| `space_3` | Spacing step 3. | 6.0 |
| `space_4` | Spacing step 4: the default gap between panels. | 8.0 |
| `space_5` | Spacing step 5. | 12.0 |
| `space_6` | Spacing step 6. | 16.0 |
| `space_7` | Spacing step 7. | 24.0 |
| `space_8` | Spacing step 8. | 32.0 |
| `border` | Border width of controls and dividers. | 1.0 |
| `focus_ring_width` | Width of the keyboard focus ring. | 2.0 |
| `focus_ring_offset` | Gap between a control and its focus ring. | 2.0 |
| `control_height` | Height of buttons, fields, sliders. | 28.0 |
| `control_height_sm` | Height of a compact control (toolbars, property rows). | 24.0 |
| `row_height` | Height of a list or tree row. | 30.0 |
| `control_pad_x` | Horizontal padding inside buttons and fields. | 12.0 |
| `icon` | Icon size in buttons. | 15.0 |
| `icon_sm` | Icon size in compact controls. | 13.0 |
| `icon_button` | Width of an icon-only button (square). | 28.0 |
| `switch_w` | Switch width. | 38.0 |
| `switch_h` | Switch height. | 22.0 |
| `switch_knob` | Switch knob diameter. | 16.0 |
| `slider_track` | Slider track thickness. | 4.0 |
| `slider_thumb` | Slider thumb diameter. | 14.0 |
| `scrollbar_zone` | Width of the scroll bar's hit zone. | 12.0 |
| `scrollbar_thin` | Scroll thumb thickness at rest. | 3.0 |
| `scrollbar_thick` | Scroll thumb thickness while hovered or dragged. | 6.0 |
| `scrollbar_inset` | Distance of the scroll thumb from the edge. | 4.0 |
| `scrollbar_min_thumb` | Shortest scroll thumb. | 28.0 |
| `caret_width` | Caret width in text fields. | 1.5 |
| `tooltip_width` | Tooltip maximum width. | 300.0 |
| `tooltip_pad` | Tooltip padding. | 10.0 |

## Text styles

Faces (`FontFace`): `regular` (Inter Regular), `semibold` (Inter SemiBold), `italic` (Inter Italic),
`mono` (JetBrains Mono Regular), `icons` (Lucide, one private-use code point per icon). The baked
atlases are in `data/fonts/*.anfont` (`[UI-03]`).

| Style | Face | Size (logical px) | Line height (× size) | Use |
|---|---|---|---|---|
| `body` | regular | 13 | 1.5 | body text |
| `small` | regular | 12 | 1.45 | property labels, list rows |
| `tiny` | regular | 11 | 1.4 | captions under values, status bar |
| `label` | semibold | 12.5 | 1.3 | control labels (buttons, tabs) |
| `caption` | semibold | 11 | 1.3 | section captions |
| `heading` | semibold | 14 | 1.3 | panel headings |
| `title` | semibold | 20 | 1.25 | dialog titles |
| `mono` | mono | 12 | 1.5 | numbers in fields, code, the console |

Icons use `TextStyle::icon(size)` (line height 1) with the `icon` / `icon_sm` sizes.

## Shadows

A shadow is the surface's rounded rectangle drawn first, moved down by `offset_y`, inset by `inset_x`
at each side, blurred by `blur` (alpha `1 - smoothstep(-blur, blur, d)` over the signed distance
`d`), in the palette's `shadow` colour times `strength`. The surface is drawn on top with a 1 px
`border_strong` border.

| Recipe | offset_y | inset_x | strength | radius | blur |
|---|---|---|---|---|---|
| `tooltip` | 6 | 0 | 1 | 10 | 20 |
| `menu` | 8 | 0 | 1 | 10 | 28 |
| `popover` | 8 | 0 | 1 | 12 | 22 |
| `notice` | 4 | 0 | 1 | 10 | 14 |
| `modal` | 16 | 0 | 1 | 16 | 48 |
| `control` | 1 | 0 | 0.5 | 8 | 3 |

## Motion

| Token | Value | Meaning |
|---|---|---|
| `ease_rate` | 16 | standard ease: `value += (target - value) * (1 - exp(-dt * rate))` per frame |
| `scroll_rate` | 22 | the same for scrolling |
| `theme_fade` | 0.25 s | theme cross-fade, smoothstep over `Theme::mix` |
| `dt_clamp` | 0.05 s | largest frame time fed to animations |
| `tooltip_delay` | 0.45 s | hover time before a tooltip |
| `caret_period` / `caret_on` | 1.0 s / 0.55 s | caret blink; restarts on every edit |
| `double_click_time` / `double_click_distance` | 0.4 s / 4 px | double and triple clicks |
| `drag_threshold` | 3 px | travel before a press becomes a drag |
| `wheel_line` | 48 px | one wheel line |
| `reduced` | false | reduced motion `[UI-09]` |

## Theme files and hot reload

A theme file is JSON. Every member is optional:

```json
{
  "base": "dark",
  "colour_neutral": false,
  "palette": { "accent": "#FF8800", "selection": "#2563EB29" },
  "metrics": { "radius": 4 },
  "text": { "body": { "face": "regular", "size": 14, "line_height": 1.5 } },
  "shadows": { "menu": { "blur": 20 } },
  "motion": { "reduced": true }
}
```

`Theme::from_json` starts from `base` (default dark) and applies the overrides;
`Theme::apply_json` applies them to an existing theme. Unknown members and token names, colours that
are not `#RGB` / `#RRGGBB` / `#RRGGBBAA`, and numbers outside 0 to 1000 are errors
(`UiError::ThemeFile`), and a failed load leaves the theme unchanged, so a debug build can watch the
file and re-apply it on every save without a typo ever being silently ignored. `Theme::to_json`
writes a complete file (every token), which reads back to the same theme up to 8-bit colour
quantisation (`theme::tests::ui_08_json_round_trip`).
