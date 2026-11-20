//! Time for the library crates: a process-wide clock that the host installs, a stopwatch, and the
//! text forms of durations and timestamps.
//!
//! `std::time::Instant` and `SystemTime` panic on `wasm32-unknown-unknown`, so nothing here uses
//! them `[PLAT-03]`. The host (the native binaries through `humanart-platform`, a web page through
//! its own shim) calls [`install_clock`] once at start-up. Until then [`now_nanos`] and
//! [`unix_millis`] return 0 and everything built on them (stopwatches, the profiler, log
//! timestamps) keeps working with zero durations: a library used without a clock loses its timings,
//! never its footing.

use std::fmt;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::collections::ArrayString;
use crate::platform::Clock;

/// A slot that holds at most one clock, set once.
struct ClockSlot(OnceLock<Arc<dyn Clock>>);

impl ClockSlot {
    const fn new() -> Self {
        Self(OnceLock::new())
    }

    fn install(&self, clock: Arc<dyn Clock>) -> bool {
        self.0.set(clock).is_ok()
    }

    #[inline]
    fn now_nanos(&self) -> u64 {
        self.0.get().map_or(0, |clock| clock.now_nanos())
    }

    #[inline]
    fn unix_millis(&self) -> u64 {
        self.0.get().map_or(0, |clock| clock.unix_millis())
    }
}

static CLOCK: ClockSlot = ClockSlot::new();

/// Installs the clock of this process. Returns `false`, changing nothing, when one is already
/// installed: the first clock stays for the life of the process, so intervals are never measured
/// across two time bases.
pub fn install_clock(clock: Arc<dyn Clock>) -> bool {
    CLOCK.install(clock)
}

/// `true` once [`install_clock`] has succeeded.
pub fn clock_installed() -> bool {
    CLOCK.0.get().is_some()
}

/// Monotonic nanoseconds from the installed clock; 0 while none is installed.
#[inline]
pub fn now_nanos() -> u64 {
    CLOCK.now_nanos()
}

/// Wall-clock Unix time in milliseconds from the installed clock; 0 while none is installed.
#[inline]
pub fn unix_millis() -> u64 {
    CLOCK.unix_millis()
}

/// Measures an interval on the installed clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stopwatch {
    start: u64,
}

impl Stopwatch {
    /// A stopwatch started now.
    #[must_use]
    pub fn start() -> Self {
        Self { start: now_nanos() }
    }

    /// A stopwatch started at the monotonic time `nanos` (for code that already read the clock).
    #[must_use]
    pub const fn started_at(nanos: u64) -> Self {
        Self { start: nanos }
    }

    /// The monotonic time at which the stopwatch was started.
    pub const fn start_nanos(&self) -> u64 {
        self.start
    }

    /// Nanoseconds from the start to the monotonic time `now` (0 if `now` is earlier).
    pub const fn elapsed_nanos_at(&self, now: u64) -> u64 {
        now.saturating_sub(self.start)
    }

    /// Nanoseconds since the start.
    pub fn elapsed_nanos(&self) -> u64 {
        self.elapsed_nanos_at(now_nanos())
    }

    /// The time since the start.
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.elapsed_nanos())
    }

    /// Seconds since the start.
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_nanos() as f64 * 1e-9
    }

    /// Returns the nanoseconds since the start and starts again from now.
    pub fn lap(&mut self) -> u64 {
        let now = now_nanos();
        let elapsed = self.elapsed_nanos_at(now);
        self.start = now;
        elapsed
    }
}

/// A duration in nanoseconds that displays itself in the unit a person would pick:
/// `870 ns`, `12.3 µs`, `4.56 ms`, `1.25 s`, `3 m 07 s`, `2 h 05 m`.
///
/// Formatting does not allocate, so it can be used in log statements and overlays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DurationText(pub u64);

impl fmt::Display for DurationText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0;
        if n < 1_000 {
            return write!(f, "{n} ns");
        }
        // Each tier rounds to its last shown digit first, so a value never prints as "1000.0 µs".
        let tenth_micros = n.saturating_add(50) / 100;
        if tenth_micros < 10_000 {
            return write!(f, "{}.{} µs", tenth_micros / 10, tenth_micros % 10);
        }
        let centi_millis = n.saturating_add(5_000) / 10_000;
        if centi_millis < 100_000 {
            return write!(f, "{}.{:02} ms", centi_millis / 100, centi_millis % 100);
        }
        let centi_seconds = n.saturating_add(5_000_000) / 10_000_000;
        if centi_seconds < 6_000 {
            return write!(f, "{}.{:02} s", centi_seconds / 100, centi_seconds % 100);
        }
        let seconds = n.saturating_add(500_000_000) / 1_000_000_000;
        if seconds < 3_600 {
            return write!(f, "{} m {:02} s", seconds / 60, seconds % 60);
        }
        let minutes = (seconds + 30) / 60;
        write!(f, "{} h {:02} m", minutes / 60, minutes % 60)
    }
}

/// [`DurationText`] as a string on the stack.
pub fn format_duration(nanos: u64) -> ArrayString<24> {
    use fmt::Write as _;
    let mut text = ArrayString::new();
    // An `ArrayString` never reports an error; the longest output is 19 bytes.
    let _ = write!(text, "{}", DurationText(nanos));
    text
}

/// A moment of wall-clock time split into calendar fields (UTC, proleptic Gregorian calendar).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CivilTime {
    /// The year (1970 or later).
    pub year: u64,
    /// The month, 1 to 12.
    pub month: u8,
    /// The day of the month, 1 to 31.
    pub day: u8,
    /// The hour, 0 to 23.
    pub hour: u8,
    /// The minute, 0 to 59.
    pub minute: u8,
    /// The second, 0 to 59 (Unix time has no leap seconds).
    pub second: u8,
    /// The millisecond, 0 to 999.
    pub millis: u16,
}

impl CivilTime {
    /// The calendar fields of a Unix time in milliseconds.
    #[must_use]
    pub const fn from_unix_millis(unix_millis: u64) -> Self {
        let millis = (unix_millis % 1_000) as u16;
        let seconds = unix_millis / 1_000;
        let (days, rest) = (seconds / 86_400, seconds % 86_400);
        // Days to civil date, after Howard Hinnant's `civil_from_days`, for non-negative days.
        let z = days + 719_468;
        let (era, day_of_era) = (z / 146_097, z % 146_097);
        let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let shifted_month = (5 * day_of_year + 2) / 153;
        let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u8;
        let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 } as u8;
        let year = year_of_era + era * 400 + if month <= 2 { 1 } else { 0 };
        Self {
            year,
            month,
            day,
            hour: (rest / 3_600) as u8,
            minute: (rest % 3_600 / 60) as u8,
            second: (rest % 60) as u8,
            millis,
        }
    }
}

/// A Unix time in milliseconds that displays as ISO 8601 UTC: `2026-11-19T10:23:45.678Z`.
///
/// Formatting does not allocate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct UtcTimestamp(pub u64);

impl fmt::Display for UtcTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = CivilTime::from_unix_millis(self.0);
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            t.year, t.month, t.day, t.hour, t.minute, t.second, t.millis
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{count_in, is_installed};
    use crate::platform::ManualClock;

    /// The clock the unit tests of this crate share: installed on first use, advancing 1 µs per
    /// reading so intervals are never zero. Tests that need exact times use their own `ClockSlot`
    /// or pass times explicitly.
    pub(crate) fn test_clock() -> &'static Arc<ManualClock> {
        static SHARED: OnceLock<Arc<ManualClock>> = OnceLock::new();
        SHARED.get_or_init(|| {
            let clock = Arc::new(ManualClock::new(1_800_000_000_000));
            clock.set_auto_advance(1_000);
            assert!(install_clock(clock.clone()), "only test_clock installs a clock in this test binary");
            clock
        })
    }

    #[test]
    fn without_a_clock_time_is_zero_and_nothing_panics() {
        let slot = ClockSlot::new();
        assert_eq!((slot.now_nanos(), slot.unix_millis()), (0, 0));
        let clock = Arc::new(ManualClock::new(77));
        clock.advance_nanos(5);
        assert!(slot.install(clock.clone()));
        assert_eq!((slot.now_nanos(), slot.unix_millis()), (5, 77));
        // The first clock stays.
        assert!(!slot.install(Arc::new(ManualClock::new(1))));
        clock.advance_millis(2);
        assert_eq!((slot.now_nanos(), slot.unix_millis()), (2_000_005, 79));
    }

    #[test]
    fn the_installed_clock_drives_the_global_functions() {
        test_clock();
        assert!(clock_installed());
        assert!(!install_clock(Arc::new(ManualClock::new(0))), "a second install is refused");
        let a = now_nanos();
        let b = now_nanos();
        assert!(b > a, "the shared test clock advances on every reading");
        assert!(unix_millis() >= 1_800_000_000_000);

        let mut watch = Stopwatch::start();
        let first = watch.elapsed_nanos();
        assert!(first >= 1_000);
        assert!(watch.elapsed() >= Duration::from_nanos(first));
        assert!(watch.elapsed_seconds() > 0.0);
        let lap = watch.lap();
        assert!(lap > first && watch.start_nanos() > a);
        assert!(watch.elapsed_nanos() < lap + 1_000_000_000);
    }

    #[test]
    fn stopwatch_arithmetic_saturates() {
        let watch = Stopwatch::started_at(1_000);
        assert_eq!(watch.start_nanos(), 1_000);
        assert_eq!(watch.elapsed_nanos_at(1_750), 750);
        assert_eq!(watch.elapsed_nanos_at(999), 0, "a clock that went backwards gives 0, not a wrap");
        assert_eq!(watch.elapsed_nanos_at(u64::MAX), u64::MAX - 1_000);
    }

    #[test]
    fn durations_pick_a_readable_unit() {
        for (nanos, expect) in [
            (0, "0 ns"),
            (1, "1 ns"),
            (999, "999 ns"),
            (1_000, "1.0 µs"),
            (12_345, "12.3 µs"),
            (999_949, "999.9 µs"),
            (999_950, "1.00 ms"),
            (1_000_000, "1.00 ms"),
            (4_564_999, "4.56 ms"),
            (16_666_667, "16.67 ms"),
            (999_994_999, "999.99 ms"),
            (999_995_000, "1.00 s"),
            (1_250_000_000, "1.25 s"),
            (59_994_999_999, "59.99 s"),
            (59_995_000_000, "1 m 00 s"),
            (187_000_000_000, "3 m 07 s"),
            (3_599_499_999_999, "59 m 59 s"),
            (3_599_500_000_000, "1 h 00 m"),
            (7_500_000_000_000, "2 h 05 m"),
            (u64::MAX, "5124095 h 35 m"),
        ] {
            assert_eq!(DurationText(nanos).to_string(), expect, "{nanos}");
            assert_eq!(format_duration(nanos), expect);
        }
    }

    // [ERR-04] Timestamps and durations go into log lines without touching the heap.
    #[test]
    fn formatting_does_not_allocate() {
        use fmt::Write as _;
        let mut text = ArrayString::<64>::new();
        let stats = count_in(|| {
            let _ = write!(text, "{} {}", UtcTimestamp(1_795_083_825_678), DurationText(4_560_000));
            std::hint::black_box(format_duration(123_456_789));
        });
        assert!(is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert_eq!(text, "2026-11-19T10:23:45.678Z 4.56 ms");
    }

    #[test]
    fn civil_time_matches_known_dates() {
        for (millis, expect) in [
            (0, "1970-01-01T00:00:00.000Z"),
            (86_399_999, "1970-01-01T23:59:59.999Z"),
            (86_400_000, "1970-01-02T00:00:00.000Z"),
            (951_782_400_000, "2000-02-29T00:00:00.000Z"),
            (951_868_800_000, "2000-03-01T00:00:00.000Z"),
            (1_078_099_199_000, "2004-02-29T23:59:59.000Z"),
            (1_234_567_890_123, "2009-02-13T23:31:30.123Z"),
            (1_709_164_800_000, "2024-02-29T00:00:00.000Z"),
            (1_735_689_599_999, "2024-12-31T23:59:59.999Z"),
            (1_735_689_600_000, "2025-01-01T00:00:00.000Z"),
            (4_102_444_800_000, "2100-01-01T00:00:00.000Z"),
            (4_107_542_400_000, "2100-03-01T00:00:00.000Z"),
            (253_402_300_799_999, "9999-12-31T23:59:59.999Z"),
        ] {
            assert_eq!(UtcTimestamp(millis).to_string(), expect, "{millis}");
        }
        let t = CivilTime::from_unix_millis(1_234_567_890_123);
        assert_eq!((t.year, t.month, t.day, t.hour, t.minute, t.second, t.millis), (2009, 2, 13, 23, 31, 30, 123));
        // No overflow at the end of the range.
        let end = CivilTime::from_unix_millis(u64::MAX);
        assert!(end.year > 500_000_000 && (1..=12).contains(&end.month) && (1..=31).contains(&end.day));
    }

    // Every day from 1970 to 2400 follows the previous one: the calendar never skips or repeats.
    #[test]
    fn civil_time_walks_the_calendar_day_by_day() {
        let month_days = |year: u64, month: u8| -> u8 {
            let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
            match month {
                2 if leap => 29,
                2 => 28,
                4 | 6 | 9 | 11 => 30,
                _ => 31,
            }
        };
        let (mut year, mut month, mut day) = (1970_u64, 1_u8, 1_u8);
        for days in 0..157_000_u64 {
            let t = CivilTime::from_unix_millis(days * 86_400_000 + 43_200_000);
            assert_eq!((t.year, t.month, t.day, t.hour), (year, month, day, 12), "day {days}");
            day += 1;
            if day > month_days(year, month) {
                day = 1;
                month += 1;
                if month > 12 {
                    month = 1;
                    year += 1;
                }
            }
        }
        assert_eq!(year, 2399);
    }
}
