//! The deterministic mutation-fuzz harness every parser's tests use `[ROB-01]`.
//!
//! A fuzz test hands [`run`] a seed corpus and a closure that feeds bytes to a parser. The harness
//! mutates the corpus with [`mutate`], driven by [`Rng`] from a fixed seed so a failure reproduces,
//! and runs the whole batch under a watchdog: a parser that loops forever fails the test instead of
//! hanging it. A panic inside the parser fails the test with the case number, which with the seed is
//! enough to replay it.
//!
//! The harness is test infrastructure for native targets; it is compiled into the library (not behind
//! `cfg(test)`) so that every crate's tests can share it.

use crate::rng::Rng;

/// Values that sit on the edges parsers get wrong: zero, sign bits, maxima.
const INTERESTING_8: [u8; 9] = [0, 1, 0x7F, 0x80, 0xFF, 0x10, 0x20, 0x40, 0xFE];
const INTERESTING_16: [u16; 8] = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x00FF, 0x0100, 0xFFFE];
const INTERESTING_32: [u32; 8] = [0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF, 0x0001_0000, 0x00FF_FFFF, 0xFFFF_FFFE];

/// Returns a mutated copy of `input`. `other` is another corpus entry to splice from (may be empty).
///
/// One to four mutations are applied, chosen from: bit flip, random byte, interesting 8/16/32-bit
/// value in either byte order, truncation, range deletion, range duplication, random insertion and a
/// splice with `other`. The result is never longer than `max_len`.
pub fn mutate(input: &[u8], other: &[u8], rng: &mut Rng, max_len: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for _ in 0..=rng.index(4) {
        match rng.index(10) {
            0 if !out.is_empty() => {
                let i = rng.index(out.len());
                out[i] ^= 1 << rng.index(8);
            }
            1 if !out.is_empty() => {
                let i = rng.index(out.len());
                out[i] = rng.next_u32() as u8;
            }
            2 if !out.is_empty() => {
                let i = rng.index(out.len());
                out[i] = INTERESTING_8[rng.index(INTERESTING_8.len())];
            }
            3 if out.len() >= 2 => {
                let i = rng.index(out.len() - 1);
                let v = INTERESTING_16[rng.index(INTERESTING_16.len())];
                let bytes = if rng.chance(0.5) { v.to_le_bytes() } else { v.to_be_bytes() };
                out[i..i + 2].copy_from_slice(&bytes);
            }
            4 if out.len() >= 4 => {
                let i = rng.index(out.len() - 3);
                let v = INTERESTING_32[rng.index(INTERESTING_32.len())];
                let bytes = if rng.chance(0.5) { v.to_le_bytes() } else { v.to_be_bytes() };
                out[i..i + 4].copy_from_slice(&bytes);
            }
            5 if !out.is_empty() => {
                let keep = rng.index(out.len());
                out.truncate(keep);
            }
            6 if !out.is_empty() => {
                let start = rng.index(out.len());
                let len = 1 + rng.index((out.len() - start).min(64));
                out.drain(start..start + len);
            }
            7 if !out.is_empty() => {
                let start = rng.index(out.len());
                let len = 1 + rng.index((out.len() - start).min(64));
                let chunk = out[start..start + len].to_vec();
                let at = rng.index(out.len() + 1);
                out.splice(at..at, chunk);
            }
            8 => {
                let len = 1 + rng.index(16);
                let at = rng.index(out.len() + 1);
                let mut chunk = vec![0_u8; len];
                rng.fill(&mut chunk);
                out.splice(at..at, chunk);
            }
            9 if !other.is_empty() && !out.is_empty() => {
                let cut = rng.index(out.len());
                let from = rng.index(other.len());
                out.truncate(cut);
                out.extend_from_slice(&other[from..]);
            }
            _ => {}
        }
    }
    out.truncate(max_len);
    out
}

/// Runs `iterations` mutated variants of every entry in `corpus` through `parse`, under a watchdog.
///
/// `parse` must return normally on every input; what it returns is ignored (errors are the expected
/// outcome of most mutations). The corpus entries themselves are run first, unmutated.
///
/// # Panics
///
/// This is a test assertion: it panics when `parse` panics on some input (reporting the corpus entry
/// and the case number), and when the whole batch takes longer than `watchdog`.
#[cfg(not(target_arch = "wasm32"))]
pub fn run<F>(name: &str, corpus: &[Vec<u8>], iterations: usize, seed: u64, watchdog: std::time::Duration, parse: F)
where
    F: Fn(&[u8]) + Send + 'static,
{
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc;

    let corpus: Vec<Vec<u8>> = corpus.to_vec();
    let max_len = corpus.iter().map(Vec::len).max().unwrap_or(0) * 2 + 1024;
    let (tx, rx) = mpsc::channel::<Result<(), String>>();
    // wasm-ok: the harness is native-only test infrastructure (see the module comment).
    let worker = std::thread::Builder::new().name(format!("fuzz-{name}")).spawn(move || {
        let mut rng = Rng::new(seed);
        for (entry, bytes) in corpus.iter().enumerate() {
            if catch_unwind(AssertUnwindSafe(|| parse(bytes))).is_err() {
                let _ = tx.send(Err(format!("corpus entry {entry} panicked unmutated")));
                return;
            }
        }
        for case in 0..iterations {
            for (entry, bytes) in corpus.iter().enumerate() {
                let other = &corpus[rng.index(corpus.len())];
                let mutated = mutate(bytes, other, &mut rng, max_len);
                if catch_unwind(AssertUnwindSafe(|| parse(&mutated))).is_err() {
                    let _ = tx.send(Err(format!(
                        "panic on corpus entry {entry}, case {case}, seed {seed}, {} bytes",
                        mutated.len()
                    )));
                    return;
                }
            }
        }
        let _ = tx.send(Ok(()));
    });
    assert!(worker.is_ok(), "fuzz {name}: could not start the worker thread");
    match rx.recv_timeout(watchdog) {
        Ok(Ok(())) => {}
        Ok(Err(message)) => panic!("fuzz {name}: {message}"),
        Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
            panic!("fuzz {name}: watchdog expired after {watchdog:?} (a parser is looping, or the batch is too large)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn mutation_is_deterministic_and_bounded() {
        let input: Vec<u8> = (0..200).map(|i| i as u8).collect();
        let (mut a, mut b) = (Rng::new(1), Rng::new(1));
        let mut changed = 0;
        for _ in 0..500 {
            let x = mutate(&input, &input, &mut a, 256);
            let y = mutate(&input, &input, &mut b, 256);
            assert_eq!(x, y);
            assert!(x.len() <= 256);
            changed += u32::from(x != input);
        }
        assert!(changed > 400, "{changed}");
        // Empty input never panics and can still grow by insertion.
        let mut rng = Rng::new(2);
        assert!((0..200).any(|_| !mutate(&[], &[], &mut rng, 64).is_empty()));
    }

    #[test]
    fn run_passes_a_total_parser() {
        run("total", &[b"hello world".to_vec(), vec![]], 200, 7, Duration::from_secs(20), |bytes| {
            let _ = std::str::from_utf8(bytes);
        });
    }

    #[test]
    #[should_panic(expected = "panic on corpus entry")]
    fn run_reports_a_panicking_parser() {
        run("panics", &[vec![1, 2, 3, 4, 5, 6, 7, 8]], 2000, 7, Duration::from_secs(20), |bytes| {
            assert!(bytes.len() >= 8, "short input");
        });
    }

    #[test]
    #[should_panic(expected = "watchdog expired")]
    fn run_reports_a_hanging_parser() {
        run("hangs", &[vec![0]], 1, 7, Duration::from_millis(200), |_| {
            std::thread::sleep(Duration::from_secs(2));
        });
    }
}
