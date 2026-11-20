//! The job system `[THR-01]` `[THR-04]`: a pool of worker threads for CPU-heavy work (registration,
//! PCA, baking, LOD generation, codec work), with handles to wait for, poll, await or cancel a job,
//! and data-parallel loops.
//!
//! Threads come from a [`Spawner`] `[PLAT-03]`. When the platform has none
//! ([`Spawner::available_threads`] is 0, single-threaded wasm) the pool has no workers and every job
//! runs inline, on the calling thread, before [`Jobs::spawn`] returns; the API and the results are
//! the same, so code written against `Jobs` works on both.
//!
//! - [`Jobs::spawn`] queues a closure and returns a [`JobHandle`]. The handle can
//!   [`wait`](JobHandle::wait), be polled once per frame with [`try_take`](JobHandle::try_take), be awaited
//!   (it is a [`Future`]), or [`cancel`](JobHandle::cancel) a job that has not started.
//! - [`Jobs::parallel_for`] and [`Jobs::parallel_map`] split `0..len` into chunks that the workers
//!   and the calling thread work through together; they return when every chunk is done.
//! - A job that panics yields [`JobError::Panicked`]; the worker survives.
//! - Dropping [`Jobs`] cancels the jobs that have not started ([`JobError::Cancelled`]) and waits
//!   for the running ones to finish.
//!
//! **Waiting never deadlocks on queued work.** A thread that waits (on a handle, on
//! [`Jobs::wait_idle`], or in a parallel loop) runs queued jobs itself until the queue is empty, so
//! jobs may spawn and wait for other jobs, and parallel loops may nest, even with one worker.
//!
//! **Lifetimes.** Closures are `'static`: a job shares data through `Arc` and returns its results.
//! Lending borrowed slices to worker threads (as `std::thread::scope` does) needs `unsafe` lifetime
//! erasure, which this crate allows only in `alloc` `[UNS-01]`.
//!
//! **Allocation.** A spawn costs two small allocations (the job box and its result slot); a parallel
//! loop costs one shared block plus one box per helping worker. The job system is for work measured
//! in tens of microseconds and up, not for per-element tasks.

use std::any::Any;
use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use crate::error::{JobError, PlatformError};
use crate::id::Sequence;
use crate::platform::Spawner;

/// The most worker threads a pool starts.
pub const MAX_WORKERS: usize = 64;

/// Queue slots reserved up front, so the queue does not grow in normal use.
const QUEUE_CAPACITY: usize = 256;

type Job = Box<dyn FnOnce() + Send + 'static>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Jobs run outside every lock of this module and their panics are caught, so a poisoned lock
    // still guards consistent data.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar.wait(guard).unwrap_or_else(PoisonError::into_inner)
}

/// The text of a panic payload.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panic with a non-text payload".to_owned()
    }
}

struct Queue {
    jobs: VecDeque<Job>,
    /// Jobs being run right now, by workers or by waiting threads.
    active: usize,
    live_workers: usize,
    shutdown: bool,
}

struct Shared {
    /// Tells this pool's workers apart from other threads ([`WORKER_OF`]).
    id: u64,
    queue: Mutex<Queue>,
    /// Signalled when a job is queued or the pool shuts down.
    work: Condvar,
    /// Signalled when the pool becomes idle or a worker exits.
    idle: Condvar,
}

static POOL_IDS: Sequence = Sequence::new();

thread_local! {
    /// The id of the pool this thread is a worker of; 0 on other threads.
    static WORKER_OF: Cell<u64> = const { Cell::new(0) };
}

impl Shared {
    fn is_own_worker(&self) -> bool {
        WORKER_OF.with(Cell::get) == self.id
    }

    /// Runs one queued job on this thread. Returns `false` when the queue was empty.
    fn run_one(&self) -> bool {
        let job = {
            let mut queue = lock(&self.queue);
            let Some(job) = queue.jobs.pop_front() else { return false };
            queue.active += 1;
            job
        };
        self.run(job);
        true
    }

    fn run(&self, job: Job) {
        // Jobs catch their own panics; this only keeps a worker alive if a drop inside one panics.
        let _ = catch_unwind(AssertUnwindSafe(job));
        let mut queue = lock(&self.queue);
        queue.active -= 1;
        if queue.active == 0 && queue.jobs.is_empty() {
            self.idle.notify_all();
        }
    }

    fn worker_loop(&self) {
        WORKER_OF.with(|pool| pool.set(self.id));
        loop {
            let job = {
                let mut queue = lock(&self.queue);
                loop {
                    if let Some(job) = queue.jobs.pop_front() {
                        queue.active += 1;
                        break Some(job);
                    }
                    if queue.shutdown {
                        break None;
                    }
                    queue = wait(&self.work, queue);
                }
            };
            match job {
                Some(job) => self.run(job),
                None => break,
            }
        }
        let mut queue = lock(&self.queue);
        queue.live_workers -= 1;
        self.idle.notify_all();
    }
}

/// A pool of worker threads; see the module documentation.
pub struct Jobs {
    shared: Arc<Shared>,
    workers: usize,
}

impl fmt::Debug for Jobs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Jobs").field("workers", &self.workers).field("queued", &self.queued()).finish_non_exhaustive()
    }
}

impl Jobs {
    /// A pool with one worker per hardware thread except the caller's (at least one, at most
    /// `max_workers` and [`MAX_WORKERS`]), or no workers when `spawner` has no threads or
    /// `max_workers` is 0.
    ///
    /// `spawner` must start a real thread for every `spawn` when it reports threads. If starting a
    /// worker fails, the workers already started are stopped and the error is returned; the caller
    /// can fall back to [`Jobs::inline`].
    pub fn new(spawner: &dyn Spawner, max_workers: usize) -> Result<Self, PlatformError> {
        let available = spawner.available_threads();
        let wanted =
            if available == 0 { 0 } else { available.saturating_sub(1).max(1) }.min(max_workers).min(MAX_WORKERS);
        let mut jobs = Self::with_shared(0);
        for index in 0..wanted {
            lock(&jobs.shared.queue).live_workers += 1;
            let shared = Arc::clone(&jobs.shared);
            let started = spawner.spawn(&format!("humanart-job-{index}"), Box::new(move || shared.worker_loop()));
            if let Err(error) = started {
                lock(&jobs.shared.queue).live_workers -= 1;
                // Dropping `jobs` stops the workers already running.
                return Err(error);
            }
            jobs.workers += 1;
        }
        Ok(jobs)
    }

    /// A pool without workers: every job runs on the calling thread when it is spawned `[THR-04]`.
    pub fn inline() -> Self {
        Self::with_shared(0)
    }

    fn with_shared(workers: usize) -> Self {
        let queue =
            Queue { jobs: VecDeque::with_capacity(QUEUE_CAPACITY), active: 0, live_workers: 0, shutdown: false };
        let shared =
            Shared { id: POOL_IDS.next(), queue: Mutex::new(queue), work: Condvar::new(), idle: Condvar::new() };
        Self { shared: Arc::new(shared), workers }
    }

    /// The number of worker threads (0 when jobs run inline).
    pub fn worker_count(&self) -> usize {
        self.workers
    }

    /// `true` when jobs run on the calling thread.
    pub fn is_inline(&self) -> bool {
        self.workers == 0
    }

    /// `true` on a worker thread of this pool.
    pub fn is_worker_thread(&self) -> bool {
        self.shared.is_own_worker()
    }

    /// Jobs waiting in the queue (not counting running ones).
    pub fn queued(&self) -> usize {
        lock(&self.shared.queue).jobs.len()
    }

    fn push(&self, job: Job) {
        lock(&self.shared.queue).jobs.push_back(job);
        self.shared.work.notify_one();
    }

    /// Runs `work` on a worker (or right now, inline) and returns a handle to its result.
    pub fn spawn<T, F>(&self, work: F) -> JobHandle<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let slot = Arc::new(Slot::new());
        let completer = Completer(Arc::clone(&slot));
        let job = move || {
            if completer.0.start() {
                let result = catch_unwind(AssertUnwindSafe(work))
                    .map_err(|payload| JobError::Panicked(panic_message(&*payload)));
                completer.0.finish(result);
            }
        };
        if self.workers == 0 {
            job();
        } else {
            self.push(Box::new(job));
        }
        JobHandle { slot, shared: Arc::clone(&self.shared) }
    }

    /// Waits until the queue is empty and no job runs, running queued jobs on this thread
    /// meanwhile. On a worker of this pool it only empties the queue (it cannot wait for itself).
    pub fn wait_idle(&self) {
        while self.shared.run_one() {}
        if self.shared.is_own_worker() {
            return;
        }
        let mut queue = lock(&self.shared.queue);
        while queue.active > 0 || !queue.jobs.is_empty() {
            queue = wait(&self.shared.idle, queue);
        }
    }

    /// Calls `work` on consecutive ranges that cover `0..len`, in parallel, and returns when all are
    /// done. Each range has at least `min_chunk` elements (except the last); the calling thread
    /// works too. Fails with the message of the first range that panicked; the others still run.
    pub fn parallel_for<F>(&self, len: usize, min_chunk: usize, work: F) -> Result<(), JobError>
    where
        F: Fn(Range<usize>) + Send + Sync + 'static,
    {
        self.parallel_map(len, min_chunk, work).map(|_| ())
    }

    /// Like [`Jobs::parallel_for`], and returns what `work` returned for each range, in range order.
    pub fn parallel_map<T, F>(&self, len: usize, min_chunk: usize, work: F) -> Result<Vec<T>, JobError>
    where
        T: Send + 'static,
        F: Fn(Range<usize>) -> T + Send + Sync + 'static,
    {
        if len == 0 {
            return Ok(Vec::new());
        }
        // A few chunks per thread balance uneven work without making chunks tiny.
        let parts = (self.workers + 1).saturating_mul(4);
        let chunk = len.div_ceil(parts).max(min_chunk).max(1);
        let chunks = len.div_ceil(chunk);
        let progress = Progress { done: 0, results: (0..chunks).map(|_| None).collect(), panic: None };
        let state = Arc::new(ParallelState {
            work,
            len,
            chunk,
            chunks,
            next: AtomicUsize::new(0),
            progress: Mutex::new(progress),
            finished: Condvar::new(),
        });
        for _ in 0..self.workers.min(chunks - 1) {
            let state = Arc::clone(&state);
            self.push(Box::new(move || state.run()));
        }
        state.run();
        let mut progress = lock(&state.progress);
        while progress.done < chunks {
            progress = wait(&state.finished, progress);
        }
        if let Some(message) = progress.panic.take() {
            return Err(JobError::Panicked(message));
        }
        // Every range stored its value before counting itself done.
        progress.results.drain(..).map(|value| value.ok_or(JobError::Cancelled)).collect()
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        let dropped = {
            let mut queue = lock(&self.shared.queue);
            queue.shutdown = true;
            std::mem::take(&mut queue.jobs)
        };
        self.shared.work.notify_all();
        // Outside the lock: dropping a job that never ran marks its handle cancelled and wakes it.
        drop(dropped);
        if self.shared.is_own_worker() {
            return;
        }
        let mut queue = lock(&self.shared.queue);
        while queue.live_workers > 0 {
            queue = wait(&self.shared.idle, queue);
        }
    }
}

struct Progress<T> {
    done: usize,
    results: Vec<Option<T>>,
    panic: Option<String>,
}

struct ParallelState<T, F> {
    work: F,
    len: usize,
    chunk: usize,
    chunks: usize,
    next: AtomicUsize,
    progress: Mutex<Progress<T>>,
    finished: Condvar,
}

impl<T, F: Fn(Range<usize>) -> T> ParallelState<T, F> {
    /// Takes ranges until none are left.
    fn run(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            if index >= self.chunks {
                return;
            }
            let start = index.saturating_mul(self.chunk);
            let range = start..start.saturating_add(self.chunk).min(self.len);
            let outcome = catch_unwind(AssertUnwindSafe(|| (self.work)(range)));
            let mut progress = lock(&self.progress);
            match outcome {
                Ok(value) => {
                    if let Some(slot) = progress.results.get_mut(index) {
                        *slot = Some(value);
                    }
                }
                Err(payload) => {
                    if progress.panic.is_none() {
                        progress.panic = Some(panic_message(&*payload));
                    }
                }
            }
            progress.done += 1;
            if progress.done == self.chunks {
                self.finished.notify_all();
            }
        }
    }
}

enum State<T> {
    Pending,
    Running,
    Done(Result<T, JobError>),
    Taken,
}

struct SlotInner<T> {
    state: State<T>,
    waker: Option<Waker>,
}

/// Where a job leaves its result.
struct Slot<T> {
    inner: Mutex<SlotInner<T>>,
    ready: Condvar,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self { inner: Mutex::new(SlotInner { state: State::Pending, waker: None }), ready: Condvar::new() }
    }

    /// Moves a pending job to running; `false` when it was cancelled.
    fn start(&self) -> bool {
        let mut inner = lock(&self.inner);
        if matches!(inner.state, State::Pending) {
            inner.state = State::Running;
            true
        } else {
            false
        }
    }

    /// Stores the result unless the job already has one.
    fn finish(&self, result: Result<T, JobError>) {
        let waker = {
            let mut inner = lock(&self.inner);
            if !matches!(inner.state, State::Pending | State::Running) {
                return;
            }
            inner.state = State::Done(result);
            inner.waker.take()
        };
        self.ready.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    /// The result if there is one, leaving `Taken` behind.
    fn take(inner: &mut SlotInner<T>) -> Option<Result<T, JobError>> {
        match std::mem::replace(&mut inner.state, State::Taken) {
            State::Done(result) => Some(result),
            State::Taken => Some(Err(JobError::AlreadyTaken)),
            other => {
                inner.state = other;
                None
            }
        }
    }
}

/// Marks the job cancelled if it is dropped without having run (queue dropped at shutdown).
struct Completer<T>(Arc<Slot<T>>);

impl<T> Drop for Completer<T> {
    fn drop(&mut self) {
        self.0.finish(Err(JobError::Cancelled));
    }
}

/// The result of a job that may not have finished yet.
///
/// Dropping the handle does not cancel the job; its result is then discarded.
pub struct JobHandle<T> {
    slot: Arc<Slot<T>>,
    shared: Arc<Shared>,
}

impl<T> fmt::Debug for JobHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobHandle").field("finished", &self.is_finished()).finish()
    }
}

impl<T> JobHandle<T> {
    /// `true` once the job has a result (or it was taken).
    pub fn is_finished(&self) -> bool {
        matches!(lock(&self.slot.inner).state, State::Done(_) | State::Taken)
    }

    /// Cancels the job if it has not started: it will never run, and its result is
    /// [`JobError::Cancelled`]. Returns `false` when the job already started or finished (a
    /// running job is not interrupted).
    pub fn cancel(&self) -> bool {
        let waker = {
            let mut inner = lock(&self.slot.inner);
            if !matches!(inner.state, State::Pending) {
                return false;
            }
            inner.state = State::Done(Err(JobError::Cancelled));
            inner.waker.take()
        };
        self.slot.ready.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
        true
    }

    /// The result, if the job has finished; for polling once per frame. After the result was
    /// taken, every call returns [`JobError::AlreadyTaken`].
    pub fn try_take(&self) -> Option<Result<T, JobError>> {
        Slot::take(&mut lock(&self.slot.inner))
    }

    /// Waits for the result, running queued jobs on this thread meanwhile (see the module
    /// documentation).
    pub fn wait(&self) -> Result<T, JobError> {
        loop {
            if let Some(result) = self.try_take() {
                return result;
            }
            if !self.shared.run_one() {
                break;
            }
        }
        // The queue is empty, so this job is running somewhere or done.
        let mut inner = lock(&self.slot.inner);
        loop {
            if let Some(result) = Slot::take(&mut inner) {
                return result;
            }
            inner = wait(&self.slot.ready, inner);
        }
    }
}

/// Awaiting a handle yields the job's result; the waker is woken when the job finishes.
impl<T> Future for JobHandle<T> {
    type Output = Result<T, JobError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = lock(&self.slot.inner);
        if let Some(result) = Slot::take(&mut inner) {
            return Poll::Ready(result);
        }
        if !inner.waker.as_ref().is_some_and(|waker| waker.will_wake(context.waker())) {
            inner.waker = Some(context.waker().clone());
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{InlineSpawner, ThreadWork};
    use std::sync::atomic::AtomicU64;
    use std::sync::mpsc;

    /// Real threads, for tests only (the library takes threads from the host).
    struct StdSpawner(usize);

    impl Spawner for StdSpawner {
        fn available_threads(&self) -> usize {
            self.0
        }

        fn spawn(&self, name: &str, work: ThreadWork) -> Result<(), PlatformError> {
            std::thread::Builder::new().name(name.to_owned()).spawn(work).map(|_| ()).map_err(PlatformError::from)
        }
    }

    /// Starts `succeed` threads, then fails.
    struct FailingSpawner {
        succeed: AtomicUsize,
        exited: Arc<AtomicUsize>,
    }

    impl Spawner for FailingSpawner {
        fn available_threads(&self) -> usize {
            8
        }

        fn spawn(&self, name: &str, work: ThreadWork) -> Result<(), PlatformError> {
            if self.succeed.fetch_sub(1, Ordering::SeqCst) == 0 {
                return Err(PlatformError::Other("no more threads".to_owned()));
            }
            let exited = Arc::clone(&self.exited);
            StdSpawner(1).spawn(
                name,
                Box::new(move || {
                    work();
                    exited.fetch_add(1, Ordering::SeqCst);
                }),
            )
        }
    }

    fn pool(workers: usize) -> Jobs {
        Jobs::new(&StdSpawner(workers + 1), workers).expect("pool")
    }

    #[test]
    fn worker_count_follows_the_platform() {
        assert_eq!(pool(3).worker_count(), 3);
        assert_eq!(Jobs::new(&StdSpawner(1), 8).expect("pool").worker_count(), 1, "at least one worker");
        assert_eq!(Jobs::new(&StdSpawner(16), 2).expect("pool").worker_count(), 2);
        assert_eq!(Jobs::new(&StdSpawner(4), 0).expect("pool").worker_count(), 0);
        let inline = Jobs::new(&InlineSpawner, 8).expect("pool");
        assert!(inline.is_inline() && !inline.is_worker_thread());
        assert!(format!("{inline:?}").contains("workers: 0"));
    }

    #[test]
    fn a_failing_spawner_stops_the_started_workers() {
        let exited = Arc::new(AtomicUsize::new(0));
        let spawner = FailingSpawner { succeed: AtomicUsize::new(2), exited: Arc::clone(&exited) };
        assert_eq!(Jobs::new(&spawner, 8).err(), Some(PlatformError::Other("no more threads".to_owned())));
        // `new` returned only after the two started workers left their loops; give them a moment to
        // run the last statement of their thread.
        let mut spins = 0;
        while exited.load(Ordering::SeqCst) < 2 && spins < 10_000 {
            std::thread::yield_now();
            spins += 1;
        }
        assert_eq!(exited.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn jobs_return_their_results() {
        let jobs = pool(4);
        let handles: Vec<_> = (0..100_u64).map(|i| jobs.spawn(move || i * i)).collect();
        let total: u64 = handles.iter().map(|h| h.wait().expect("result")).sum();
        assert_eq!(total, (0..100).map(|i| i * i).sum());
        assert_eq!(handles[3].wait(), Err(JobError::AlreadyTaken));
        assert_eq!(handles[3].try_take(), Some(Err(JobError::AlreadyTaken)));
        assert!(handles[3].is_finished());
        jobs.wait_idle();
        assert_eq!(jobs.queued(), 0);
    }

    // [THR-01] A job runs on a worker thread, not on the caller.
    #[test]
    fn jobs_run_on_worker_threads() {
        let jobs = Arc::new(pool(2));
        let inner = Arc::clone(&jobs);
        let on_worker =
            jobs.spawn(move || (inner.is_worker_thread(), std::thread::current().name().map(str::to_owned)));
        // Poll instead of `wait`, which could run the job on this thread.
        while !on_worker.is_finished() {
            std::thread::yield_now();
        }
        let (flag, name) = on_worker.wait().expect("result");
        assert!(flag && !jobs.is_worker_thread());
        assert!(name.is_some_and(|n| n.starts_with("humanart-job-")));
    }

    // [THR-04] Without threads every job runs inline, before `spawn` returns, with the same results.
    #[test]
    fn inline_mode_runs_everything_on_the_caller() {
        let jobs = Jobs::new(&InlineSpawner, 8).expect("pool");
        let caller = std::thread::current().id();
        let handle = jobs.spawn(move || std::thread::current().id() == caller);
        assert!(handle.is_finished());
        assert_eq!(handle.try_take(), Some(Ok(true)));
        assert!(!handle.cancel(), "nothing to cancel once it ran");
        let panicked = jobs.spawn(|| -> u8 { panic!("inline boom") });
        assert_eq!(panicked.wait(), Err(JobError::Panicked("inline boom".to_owned())));
        let sums = jobs.parallel_map(1_000, 1, Iterator::sum::<usize>).expect("map");
        assert_eq!(sums.iter().sum::<usize>(), 999 * 1_000 / 2);
        assert_eq!(sums.len(), 4, "one thread: four chunks");
        // Nested use works the same way.
        let jobs = Arc::new(jobs);
        let inner = Arc::clone(&jobs);
        let nested = jobs.spawn(move || inner.spawn(|| 7).wait().expect("inner") + 1);
        assert_eq!(nested.wait(), Ok(8));
        jobs.wait_idle();
    }

    #[test]
    fn a_panicking_job_reports_and_the_worker_survives() {
        let jobs = pool(1);
        let bad = jobs.spawn(|| -> u32 { panic!("boom {}", 42) });
        let odd = jobs.spawn(|| -> u32 { std::panic::panic_any(17_u8) });
        let good = jobs.spawn(|| 5_u32);
        assert_eq!(bad.wait(), Err(JobError::Panicked("boom 42".to_owned())));
        assert!(matches!(odd.wait(), Err(JobError::Panicked(m)) if m.contains("non-text")));
        assert_eq!(good.wait(), Ok(5));
        assert_eq!(jobs.spawn(|| 6_u32).wait(), Ok(6), "the worker is still there");
    }

    #[test]
    fn cancel_before_start_and_cancel_on_shutdown() {
        let jobs = pool(1);
        let (open, gate) = mpsc::channel::<()>();
        let (started_tx, started) = mpsc::channel::<()>();
        let blocker = jobs.spawn(move || {
            started_tx.send(()).expect("send");
            gate.recv().expect("gate");
            1
        });
        started.recv().expect("the worker took the blocker");
        let ran = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&ran);
        let queued = jobs.spawn(move || counter.fetch_add(1, Ordering::SeqCst));
        assert!(!queued.is_finished());
        assert!(queued.cancel());
        assert!(!queued.cancel(), "only once");
        assert_eq!(queued.wait(), Err(JobError::Cancelled));
        assert!(!blocker.cancel(), "a running job cannot be cancelled");

        let counter = Arc::clone(&ran);
        let orphan = jobs.spawn(move || counter.fetch_add(1, Ordering::SeqCst));
        open.send(()).expect("open the gate");
        // The blocker finishes; `orphan` may or may not start before the pool shuts down.
        drop(jobs);
        assert_eq!(blocker.wait(), Ok(1));
        match orphan.wait() {
            Ok(_) => assert_eq!(ran.load(Ordering::SeqCst), 1),
            Err(error) => assert_eq!((error, ran.load(Ordering::SeqCst)), (JobError::Cancelled, 0)),
        }
    }

    #[test]
    fn dropping_the_pool_cancels_queued_jobs_and_waits_for_running_ones() {
        let jobs = pool(1);
        let (started_tx, started) = mpsc::channel::<()>();
        let (open, gate) = mpsc::channel::<()>();
        let finished = Arc::new(AtomicUsize::new(0));
        let flag = Arc::clone(&finished);
        let running = jobs.spawn(move || {
            started_tx.send(()).expect("send");
            gate.recv().expect("gate");
            flag.store(1, Ordering::SeqCst);
        });
        started.recv().expect("started");
        let queued: Vec<_> = (0..10).map(|i| jobs.spawn(move || i)).collect();
        let dropper = std::thread::spawn(move || drop(jobs));
        // The only worker is blocked, so the queued jobs finish only by being cancelled by the drop.
        while !queued.iter().all(JobHandle::is_finished) {
            std::thread::yield_now();
        }
        assert!(!running.is_finished());
        open.send(()).expect("open");
        dropper.join().expect("dropper");
        assert_eq!(finished.load(Ordering::SeqCst), 1, "drop waited for the running job");
        assert_eq!(running.wait(), Ok(()));
        for handle in queued {
            assert_eq!(handle.try_take(), Some(Err(JobError::Cancelled)));
        }
    }

    // [AC-WP02-2] Stress: many jobs from many threads, jobs that spawn and wait for jobs, nested
    // parallel loops, all on two workers, without deadlock and with every result right.
    #[test]
    fn stress_nested_spawns_and_parallel_loops() {
        let jobs = Arc::new(pool(2));
        let total = Arc::new(AtomicU64::new(0));
        let producers: Vec<_> = (0..4_u64)
            .map(|p| {
                let (jobs, total) = (Arc::clone(&jobs), Arc::clone(&total));
                std::thread::spawn(move || {
                    let handles: Vec<_> = (0..2_000_u64).map(|i| jobs.spawn(move || p * 10_000 + i)).collect();
                    for handle in handles {
                        total.fetch_add(handle.wait().expect("job"), Ordering::Relaxed);
                    }
                })
            })
            .collect();
        let outer: Vec<_> = (0..50_u64)
            .map(|i| {
                let inner_jobs = Arc::clone(&jobs);
                jobs.spawn(move || {
                    let children: Vec<_> = (0..10_u64).map(|k| inner_jobs.spawn(move || i * k)).collect();
                    let sum: u64 = children.iter().map(|c| c.wait().expect("child")).sum();
                    let loop_jobs = Arc::clone(&inner_jobs);
                    let squares = inner_jobs
                        .parallel_map(100, 3, move |range| {
                            let inner = loop_jobs.parallel_map(range.len(), 1, |r| r.len()).expect("inner loop");
                            range.map(|x| (x * x) as u64).sum::<u64>() + inner.iter().sum::<usize>() as u64
                        })
                        .expect("loop");
                    sum + squares.iter().sum::<u64>()
                })
            })
            .collect();
        for producer in producers {
            producer.join().expect("producer");
        }
        let expected_producers: u64 = (0..4_u64).map(|p| (0..2_000_u64).map(|i| p * 10_000 + i).sum::<u64>()).sum();
        assert_eq!(total.load(Ordering::Relaxed), expected_producers);
        let squares: u64 = (0..100_u64).map(|x| x * x).sum::<u64>() + 100;
        for (i, handle) in outer.iter().enumerate() {
            let i = i as u64;
            assert_eq!(handle.wait(), Ok(i * 45 + squares));
        }
        jobs.wait_idle();
        assert_eq!(jobs.queued(), 0);
    }

    #[test]
    fn parallel_loops_cover_every_index_once() {
        let jobs = pool(3);
        for (len, min_chunk) in [(0, 1), (1, 1), (7, 1), (1_000, 1), (1_000, 300), (10_007, 64), (5, 0)] {
            let hits = Arc::new((0..len).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
            let seen = Arc::clone(&hits);
            jobs.parallel_for(len, min_chunk, move |range| {
                assert!(!range.is_empty());
                for i in range {
                    seen[i].fetch_add(1, Ordering::Relaxed);
                }
            })
            .expect("loop");
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "len {len}");
            let ranges = jobs.parallel_map(len, min_chunk, |range| range).expect("map");
            let mut next = 0;
            for range in &ranges {
                assert_eq!(range.start, next, "ranges come back in order");
                assert!(range.len() >= min_chunk.min(len - range.start).max(1));
                next = range.end;
            }
            assert_eq!(next, len);
        }
    }

    #[test]
    fn a_panic_in_a_parallel_loop_is_reported_after_all_ranges_ran() {
        let jobs = pool(2);
        let done = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&done);
        let result = jobs.parallel_for(100, 1, move |range| {
            counter.fetch_add(range.len(), Ordering::SeqCst);
            assert!(!range.contains(&50), "range with 50");
        });
        assert!(matches!(result, Err(JobError::Panicked(m)) if m.contains("range with 50")));
        assert_eq!(done.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn handles_are_futures() {
        use std::task::Wake;
        struct Count(AtomicUsize);
        impl Wake for Count {
            fn wake(self: Arc<Self>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let jobs = pool(1);
        let (open, gate) = mpsc::channel::<()>();
        let mut handle = jobs.spawn(move || {
            gate.recv().expect("gate");
            9
        });
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let mut context = Context::from_waker(&waker);
        assert!(Pin::new(&mut handle).poll(&mut context).is_pending());
        assert!(Pin::new(&mut handle).poll(&mut context).is_pending());
        open.send(()).expect("open");
        jobs.wait_idle();
        assert_eq!(count.0.load(Ordering::SeqCst), 1, "woken once when the job finished");
        assert_eq!(Pin::new(&mut handle).poll(&mut context), Poll::Ready(Ok(9)));
        assert!(format!("{handle:?}").contains("finished: true"));
    }
}
