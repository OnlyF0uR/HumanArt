//! Futures without an async runtime `[DEP-05]`: [`block_on`] for the few places that must wait for
//! a future (wgpu start-up, a [`JobHandle`](crate::jobs::JobHandle) at exit), and [`LocalExecutor`],
//! which the host drives once per frame for futures that finish over several frames (asynchronous
//! GPU readbacks `[PERF-R3]`, jobs awaited by the UI).
//!
//! `LocalExecutor` never blocks, so it is also how futures run on single-threaded wasm, where the
//! browser's event loop must keep running between polls `[THR-04]`.

use std::cell::RefCell;
use std::fmt;
use std::future::Future;
use std::pin::{Pin, pin};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Wake, Waker};

use crate::error::JobError;

/// Rounds of polling one [`LocalExecutor::run_until_stalled`] call does at most: a task that keeps
/// waking itself is polled this many times per call and continues on the next call.
pub const MAX_ROUNDS: usize = 64;

/// Whether this target can put a thread to sleep until a waker fires. Single-threaded wasm cannot
/// (`Condvar::wait` panics there), and nothing else could fire the waker anyway.
const CAN_BLOCK: bool = !cfg!(all(target_family = "wasm", not(target_feature = "atomics")));

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // The data behind these locks is a flag or a list of indices, consistent at every point.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The waker of [`block_on`]: a flag and a condition variable.
#[derive(Default)]
struct Signal {
    woken: Mutex<bool>,
    condvar: Condvar,
}

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        *lock(&self.woken) = true;
        self.condvar.notify_one();
    }
}

/// Runs `future` to completion on this thread, sleeping while it waits for its waker.
///
/// On a platform without threads (single-threaded wasm) a future that is pending without having
/// woken itself can never finish while this thread waits, so instead of hanging or panicking this
/// returns [`JobError::WouldBlock`]; use [`LocalExecutor`] there. Everywhere else it returns `Ok`.
pub fn block_on<F: Future>(future: F) -> Result<F::Output, JobError> {
    block_on_with(future, CAN_BLOCK)
}

fn block_on_with<F: Future>(future: F, can_block: bool) -> Result<F::Output, JobError> {
    let signal = Arc::new(Signal::default());
    let waker = Waker::from(Arc::clone(&signal));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return Ok(output);
        }
        let mut woken = lock(&signal.woken);
        if !*woken && !can_block {
            return Err(JobError::WouldBlock);
        }
        while !*woken {
            woken = signal.condvar.wait(woken).unwrap_or_else(PoisonError::into_inner);
        }
        *woken = false;
    }
}

/// The indices of tasks whose wakers fired.
struct ReadyList {
    indices: Mutex<Vec<usize>>,
}

/// The waker of one task slot of a [`LocalExecutor`].
struct TaskWaker {
    index: usize,
    /// Set while the index is in the ready list, so it is listed at most once.
    queued: AtomicBool,
    list: Arc<ReadyList>,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            // Never grows the list: it has room for every slot and each slot is listed once.
            lock(&self.list.indices).push(self.index);
        }
    }
}

type Task = Pin<Box<dyn Future<Output = ()>>>;

/// A single-threaded executor that polls its tasks only when asked, never blocking.
///
/// [`spawn`](Self::spawn) a future, then call [`run_until_stalled`](Self::run_until_stalled) once
/// per frame: it polls the tasks whose wakers fired since the last call (wakers may fire on any
/// thread). Tasks need not be `Send`. Running tasks allocates nothing; spawning allocates the task
/// box, and the slot tables grow only when more tasks are alive at once than ever before.
pub struct LocalExecutor {
    tasks: Vec<Option<Task>>,
    flags: Vec<Arc<TaskWaker>>,
    wakers: Vec<Waker>,
    free: Vec<usize>,
    ready: Arc<ReadyList>,
    scratch: Vec<usize>,
    live: usize,
}

impl fmt::Debug for LocalExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalExecutor")
            .field("live", &self.live)
            .field("slots", &self.tasks.len())
            .finish_non_exhaustive()
    }
}

impl Default for LocalExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalExecutor {
    /// An executor without tasks.
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            flags: Vec::new(),
            wakers: Vec::new(),
            free: Vec::new(),
            ready: Arc::new(ReadyList { indices: Mutex::new(Vec::new()) }),
            scratch: Vec::new(),
            live: 0,
        }
    }

    /// Adds a task; it is polled on the next [`run_until_stalled`](Self::run_until_stalled). The
    /// handle receives its output.
    pub fn spawn<F>(&mut self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + 'static,
    {
        let output = Rc::new(RefCell::new(TaskOutput { value: None, done: false }));
        let sender = Rc::clone(&output);
        let task: Task = Box::pin(async move {
            let value = future.await;
            let mut output = sender.borrow_mut();
            output.value = Some(value);
            output.done = true;
        });
        let index = if let Some(index) = self.free.pop() {
            if let Some(slot) = self.tasks.get_mut(index) {
                *slot = Some(task);
            }
            index
        } else {
            let index = self.tasks.len();
            let flag = Arc::new(TaskWaker { index, queued: AtomicBool::new(false), list: Arc::clone(&self.ready) });
            self.wakers.push(Waker::from(Arc::clone(&flag)));
            self.flags.push(flag);
            self.tasks.push(Some(task));
            // Room for every slot in all three index lists, so running never grows them.
            let slots = self.tasks.len();
            let mut ready = lock(&self.ready.indices);
            let listed = ready.len();
            ready.reserve(slots.saturating_sub(listed));
            drop(ready);
            self.scratch.reserve(slots.saturating_sub(self.scratch.len()));
            self.free.reserve(slots.saturating_sub(self.free.len()));
            index
        };
        self.live += 1;
        if let Some(waker) = self.wakers.get(index) {
            waker.wake_by_ref();
        }
        TaskHandle { output }
    }

    /// Polls every task whose waker fired, repeating while polling wakes more (at most
    /// [`MAX_ROUNDS`] rounds). Returns the number of polls.
    pub fn run_until_stalled(&mut self) -> usize {
        let mut polls = 0;
        for _ in 0..MAX_ROUNDS {
            // Swap the lists so wakers can fire (and list) while this round polls; both vectors
            // keep their capacity.
            std::mem::swap(&mut self.scratch, &mut *lock(&self.ready.indices));
            if self.scratch.is_empty() {
                break;
            }
            for position in 0..self.scratch.len() {
                let Some(&index) = self.scratch.get(position) else { break };
                let (Some(slot), Some(flag), Some(waker)) =
                    (self.tasks.get_mut(index), self.flags.get(index), self.wakers.get(index))
                else {
                    continue;
                };
                // Cleared before polling, so a wake during the poll lists the task again.
                flag.queued.store(false, Ordering::Release);
                let Some(task) = slot.as_mut() else { continue };
                polls += 1;
                if task.as_mut().poll(&mut Context::from_waker(waker)).is_ready() {
                    *slot = None;
                    self.free.push(index);
                    self.live -= 1;
                }
            }
            self.scratch.clear();
        }
        polls
    }

    /// Tasks that have not finished.
    pub fn len(&self) -> usize {
        self.live
    }

    /// `true` when every task has finished.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
}

struct TaskOutput<T> {
    value: Option<T>,
    done: bool,
}

/// The output of a task of a [`LocalExecutor`].
pub struct TaskHandle<T> {
    output: Rc<RefCell<TaskOutput<T>>>,
}

impl<T> fmt::Debug for TaskHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskHandle").field("finished", &self.is_finished()).finish()
    }
}

impl<T> TaskHandle<T> {
    /// `true` once the task has finished (even after its output was taken).
    pub fn is_finished(&self) -> bool {
        self.output.borrow().done
    }

    /// The task's output, once; `None` before the task finishes and after the output was taken.
    pub fn try_take(&self) -> Option<T> {
        self.output.borrow_mut().value.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{count_in, is_installed};
    use crate::jobs::Jobs;
    use crate::platform::InlineSpawner;
    use std::cell::Cell;

    /// Pending once (waking itself), then ready.
    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                return Poll::Ready(());
            }
            self.0 = true;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }

    /// A one-shot value set from outside, possibly from another thread.
    #[derive(Default)]
    struct Shot {
        value: Mutex<Option<u32>>,
        waker: Mutex<Option<Waker>>,
    }

    impl Shot {
        fn set(&self, value: u32) {
            *lock(&self.value) = Some(value);
            if let Some(waker) = lock(&self.waker).take() {
                waker.wake();
            }
        }
    }

    struct ShotFuture(Arc<Shot>);

    impl Future for ShotFuture {
        type Output = u32;
        fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<u32> {
            *lock(&self.0.waker) = Some(context.waker().clone());
            match lock(&self.0.value).take() {
                Some(value) => Poll::Ready(value),
                None => Poll::Pending,
            }
        }
    }

    #[test]
    fn block_on_ready_and_self_waking_futures() {
        assert_eq!(block_on(async { 3 }), Ok(3));
        assert_eq!(
            block_on(async {
                YieldOnce(false).await;
                4
            }),
            Ok(4)
        );
        // [THR-04] Without threads, a future that wakes itself still finishes.
        assert_eq!(
            block_on_with(
                async {
                    YieldOnce(false).await;
                    5
                },
                false
            ),
            Ok(5)
        );
    }

    // [THR-04] Without threads, waiting for a waker that nothing can fire is an error, not a hang.
    #[test]
    fn block_on_without_threads_reports_would_block() {
        let shot = Arc::new(Shot::default());
        assert_eq!(block_on_with(ShotFuture(Arc::clone(&shot)), false), Err(JobError::WouldBlock));
        shot.set(8);
        assert_eq!(block_on_with(ShotFuture(shot), false), Ok(8));
    }

    #[test]
    fn block_on_sleeps_until_another_thread_wakes_it() {
        let shot = Arc::new(Shot::default());
        let setter = Arc::clone(&shot);
        let thread = std::thread::spawn(move || {
            // Wait until the future has registered its waker, then fire it.
            while lock(&setter.waker).is_none() {
                std::thread::yield_now();
            }
            setter.set(42);
        });
        assert_eq!(block_on(ShotFuture(shot)), Ok(42));
        thread.join().expect("setter");
    }

    #[test]
    fn job_handles_can_be_awaited() {
        let jobs = Jobs::inline();
        assert_eq!(block_on(jobs.spawn(|| 6 * 7)), Ok(Ok(42)));
        let jobs = Jobs::new(&InlineSpawner, 4).expect("pool");
        let mut executor = LocalExecutor::new();
        let handle = executor.spawn(jobs.spawn(|| "done"));
        executor.run_until_stalled();
        assert_eq!(handle.try_take(), Some(Ok("done")));
    }

    #[test]
    fn local_executor_polls_only_woken_tasks() {
        let mut executor = LocalExecutor::new();
        assert!(executor.is_empty());
        let shots: Vec<Arc<Shot>> = (0..3).map(|_| Arc::new(Shot::default())).collect();
        let handles: Vec<_> = shots.iter().map(|shot| executor.spawn(ShotFuture(Arc::clone(shot)))).collect();
        let quick = executor.spawn(async {
            YieldOnce(false).await;
            "quick"
        });
        assert_eq!(executor.len(), 4);
        assert_eq!(executor.run_until_stalled(), 5, "four first polls and the yielded task once more");
        assert_eq!((quick.try_take(), quick.try_take()), (Some("quick"), None));
        assert!(quick.is_finished());
        assert_eq!(executor.run_until_stalled(), 0, "nothing was woken");
        shots[1].set(11);
        assert_eq!(executor.run_until_stalled(), 1);
        assert_eq!(handles[1].try_take(), Some(11));
        assert!(!handles[0].is_finished() && executor.len() == 2);
        // A wake from another thread is picked up on the next run.
        let shot = Arc::clone(&shots[2]);
        std::thread::spawn(move || shot.set(22)).join().expect("thread");
        executor.run_until_stalled();
        assert_eq!(handles[2].try_take(), Some(22));
        // Freed slots are reused.
        let again = executor.spawn(async { 1 });
        executor.run_until_stalled();
        assert_eq!(again.try_take(), Some(1));
        assert!(format!("{executor:?}").contains("slots: 4") && format!("{again:?}").contains("true"));
    }

    #[test]
    fn a_task_that_always_yields_is_bounded_per_call() {
        struct Forever;
        impl Future for Forever {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }
        let mut executor = LocalExecutor::default();
        let _handle = executor.spawn(Forever);
        assert_eq!(executor.run_until_stalled(), MAX_ROUNDS);
        assert_eq!(executor.run_until_stalled(), MAX_ROUNDS);
    }

    // [PERF-R1] Driving tasks once per frame allocates nothing.
    #[test]
    fn running_tasks_does_not_allocate() {
        struct Ticks(Rc<Cell<u32>>, Rc<Cell<bool>>);
        impl Future for Ticks {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                self.0.set(self.0.get() + 1);
                if self.1.get() {
                    return Poll::Ready(());
                }
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }
        let mut executor = LocalExecutor::new();
        let stop = Rc::new(Cell::new(false));
        let counters: Vec<Rc<Cell<u32>>> = (0..8).map(|_| Rc::new(Cell::new(0))).collect();
        let handles: Vec<_> =
            counters.iter().map(|counter| executor.spawn(Ticks(Rc::clone(counter), Rc::clone(&stop)))).collect();
        let stats = count_in(|| {
            for _frame in 0..600 {
                std::hint::black_box(executor.run_until_stalled());
            }
        });
        assert!(is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert!(counters.iter().all(|c| c.get() == 600 * MAX_ROUNDS as u32));
        // Finishing frees the task boxes but allocates nothing.
        stop.set(true);
        let stats = count_in(|| {
            std::hint::black_box(executor.run_until_stalled());
        });
        assert_eq!((stats.allocations, stats.reallocations), (0, 0), "{stats:?}");
        assert!(executor.is_empty() && handles.iter().all(TaskHandle::is_finished));
    }
}
