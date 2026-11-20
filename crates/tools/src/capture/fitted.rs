//! The result of a template-to-scan fit `[WP-25]` as a file (`.hafit`). The format and the
//! displacement field moved to [`humanart_model::scanfit::fitted`] (the generator consumes them;
//! wasm-safe) and are re-exported here; this module adds the conversion from a registration report.

pub use humanart_model::scanfit::fitted::*;

use super::register::FitReport;

/// The file summary of a registration report and its [`super::register::FitFlags`] bits.
#[must_use]
pub const fn summary(report: &FitReport, flags: u32) -> FittedSummary {
    FittedSummary {
        rms_error: report.rms_error,
        max_error: report.max_error,
        matched_fraction: report.matched_fraction,
        landmark_rms: report.landmark_rms,
        folded_fraction: report.folded_fraction,
        flags,
    }
}
