//! The custom GPU-rendered UI toolkit: layout, widgets, text, docking and input. Knows nothing of characters.
#![forbid(unsafe_code)]

pub mod bench;
pub mod color;
pub mod controls;
pub mod dock;
pub mod editors;
pub mod error;
pub mod font;
pub mod fontbake;
pub mod geom;
pub mod icons;
pub mod input;
mod kit;
pub mod layout;
pub mod paint;
pub mod popup;
pub mod render;
pub mod text;
pub mod theme;
pub mod ui;
pub mod views;
pub mod widgets;

#[cfg(test)]
mod tests;

pub use color::Color;
pub use error::UiError;
pub use geom::{Insets, Point, Rect, Size};
pub use theme::Theme;

// [PERF-R1] The unit-test binary counts allocations so the steady-state tests can assert zero.
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;
