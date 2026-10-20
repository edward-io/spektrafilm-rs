//! Cooperative admission for image work. CPU and GPU have separate lanes so
//! decoding can overlap rendering. GPU stages keep their buffers while yielding.
//!
//! Foreground work (drag frames, then refine renders and photo decodes) has its
//! own CPU slots, so it never waits behind background work. Background work also
//! runs its parallel loops in a smaller, lower-priority thread pool (see
//! [`Task::run`]) so it doesn't take cores from the foreground once admitted.
use std::{
    cell::RefCell,
    collections::BTreeSet,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

/// Most urgent first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Low-resolution frames while a control is being dragged.
    Drag,
    /// The main image: full-resolution refines, decoding a newly selected photo.
    Interactive,
    /// Short work for something on screen, e.g. catalog thumbnails.
    Visible,
    Export,
    /// Speculative work such as preloading neighbouring photos.
    Background,
}

impl Priority {
    pub fn is_foreground(self) -> bool {
        self <= Priority::Interactive
    }
}

/// Set once a scheduler exists, so unscheduled GPU work can be flagged.
static IN_USE: AtomicBool = AtomicBool::new(false);

#[derive(Clone)]
pub struct Scheduler {
    cpu: Arc<Lane>,
    gpu: Arc<Lane>,
}

impl Default for Scheduler {
    fn default() -> Self {
        IN_USE.store(true, Ordering::Relaxed);
        Self {
            cpu: Arc::new(Lane::new(Kind::Cpu)),
            gpu: Arc::new(Lane::new(Kind::Gpu)),
        }
    }
}

impl Scheduler {
    pub fn task(&self, priority: Priority) -> Task {
        Task {
            scheduler: self.clone(),
            priority,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Clone)]
pub struct Task {
    scheduler: Scheduler,
    priority: Priority,
    cancelled: Arc<AtomicBool>,
}

#[derive(Debug, thiserror::Error)]
#[error("processing cancelled")]
pub struct Cancelled;

impl Task {
    pub fn priority(&self) -> Priority {
        self.priority
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.scheduler.cpu.wake();
        self.scheduler.gpu.wake();
    }

    pub fn check(&self) -> Result<(), Cancelled> {
        if self.cancelled.load(Ordering::Relaxed) {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn cpu(&self) -> Result<Permit, Cancelled> {
        self.scheduler
            .cpu
            .acquire(self.priority, Some(self.cancelled.clone()))
    }

    pub fn enter(&self) -> Scope {
        Scope {
            previous: CURRENT.with(|current| current.replace(Some(self.clone()))),
            thread: PhantomData,
        }
    }

    /// Run `f` inside the task. Background work runs on the low-priority
    /// thread pool, so its parallel loops use those threads instead of the
    /// global pool the foreground uses.
    pub fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        let run = || {
            let _scope = self.enter();
            f()
        };
        if self.priority.is_foreground() {
            run()
        } else {
            background_pool().install(run)
        }
    }
}

/// Fewer threads than cores, at a lower OS priority where available (on macOS
/// the utility QoS class, which also keeps it off the performance cores the
/// UI and foreground renders want).
fn background_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let threads = if cfg!(target_os = "macos") {
            cores
        } else {
            (cores / 2).max(1)
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("spektrafilm-background-{i}"))
            .start_handler(|_| lower_thread_priority())
            .build()
            .expect("background thread pool")
    })
}

#[cfg(target_os = "macos")]
fn lower_thread_priority() {
    const QOS_CLASS_UTILITY: u32 = 0x11;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: only changes the scheduling class of the calling thread.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn lower_thread_priority() {}

thread_local! {
    static CURRENT: RefCell<Option<Task>> = const { RefCell::new(None) };
}

#[must_use = "keep the scope alive while processing the task"]
pub struct Scope {
    previous: Option<Task>,
    thread: PhantomData<Rc<()>>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|current| current.replace(self.previous.take()));
    }
}

fn current() -> Option<Task> {
    CURRENT.with(|current| current.borrow().clone())
}

pub fn cpu_permit() -> Result<Option<Permit>, Cancelled> {
    current().as_ref().map(Task::cpu).transpose()
}

/// Whether the current task (if any) has been cancelled.
pub fn check() -> Result<(), Cancelled> {
    current().as_ref().map_or(Ok(()), Task::check)
}

/// GPU admission for short kernels. They complete their calculation; callers
/// discard cancelled results.
pub fn gpu_permit() -> Option<Permit> {
    let Some(task) = current() else {
        warn_unscheduled();
        return None;
    };
    Some(
        task.scheduler
            .gpu
            .acquire(task.priority, None)
            .expect("uncancellable GPU admission"),
    )
}

/// GPU admission for long chains: gives up if the task is cancelled while
/// waiting, and the permit stays cancellable when it yields.
pub fn gpu_permit_cancellable() -> Result<Option<Permit>, Cancelled> {
    let Some(task) = current() else {
        warn_unscheduled();
        return Ok(None);
    };
    task.scheduler
        .gpu
        .acquire(task.priority, Some(task.cancelled.clone()))
        .map(Some)
}

/// The task context is per thread: work fanned out to rayon workers runs
/// without it, unscheduled. Flag that in debug builds.
fn warn_unscheduled() {
    if cfg!(debug_assertions)
        && IN_USE.load(Ordering::Relaxed)
        && rayon::current_thread_index().is_some()
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                "GPU work on a worker thread without a scheduler task; enter the task on that thread"
            )
        });
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Cpu,
    Gpu,
}

/// CPU slots per pool.
const FOREGROUND_SLOTS: usize = 2;
const BACKGROUND_SLOTS: usize = 2;

struct Lane {
    kind: Kind,
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Default)]
struct State {
    /// Admitted permits per priority.
    active: [usize; 5],
    next: u64,
    waiting: BTreeSet<(Priority, u64)>,
}

impl State {
    fn count(&self, priorities: &[Priority]) -> usize {
        priorities.iter().map(|&p| self.active[p as usize]).sum()
    }
}

impl Lane {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        }
    }

    fn wake(&self) {
        let _state = self.state.lock().unwrap();
        self.changed.notify_all();
    }

    /// Tickets that compete with `priority` for the same slots. On the CPU
    /// foreground and background work have separate slots; the GPU is shared.
    fn same_pool(&self, a: Priority, b: Priority) -> bool {
        self.kind == Kind::Gpu || a.is_foreground() == b.is_foreground()
    }

    fn has_room(&self, state: &State, priority: Priority) -> bool {
        use Priority::*;
        let foreground = state.count(&[Drag, Interactive]);
        if self.kind == Kind::Gpu {
            return state.active.iter().sum::<usize>() == 0;
        }
        match priority {
            Drag | Interactive => foreground < FOREGROUND_SLOTS,
            // Visible work can use both background slots, so a long export
            // never holds up thumbnails.
            Visible => state.count(&[Visible, Export, Background]) < BACKGROUND_SLOTS,
            // Exports and speculative work share one slot, and speculative work
            // also waits for the foreground to finish.
            Export | Background => {
                state.count(&[Visible, Export, Background]) < BACKGROUND_SLOTS
                    && state.count(&[Export, Background]) == 0
                    && (priority == Export || foreground == 0)
            }
        }
    }

    fn acquire(
        self: &Arc<Self>,
        priority: Priority,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<Permit, Cancelled> {
        let mut state = self.state.lock().unwrap();
        let ticket = (priority, state.next);
        state.next += 1;
        state.waiting.insert(ticket);
        self.changed.notify_all();
        loop {
            if cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                state.waiting.remove(&ticket);
                self.changed.notify_all();
                return Err(Cancelled);
            }
            let first_in_pool = state
                .waiting
                .iter()
                .find(|(p, _)| self.same_pool(*p, priority))
                == Some(&ticket);
            if first_in_pool && self.has_room(&state, priority) {
                state.waiting.remove(&ticket);
                state.active[priority as usize] += 1;
                self.changed.notify_all();
                return Ok(Permit {
                    lane: self.clone(),
                    priority,
                    cancel,
                    held: true,
                });
            }
            state = self.changed.wait(state).unwrap();
        }
    }
}

#[must_use = "keep the permit alive while using processing capacity"]
pub struct Permit {
    lane: Arc<Lane>,
    priority: Priority,
    cancel: Option<Arc<AtomicBool>>,
    held: bool,
}

impl Permit {
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Drag frames are small and run in one go; everything else stops at
    /// stage boundaries so more urgent work can go first.
    pub fn is_preemptible(&self) -> bool {
        self.priority != Priority::Drag
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    /// More urgent work is waiting for this permit's slot.
    pub fn should_yield(&self) -> bool {
        let state = self.lane.state.lock().unwrap();
        state
            .waiting
            .iter()
            .find(|(p, _)| self.lane.same_pool(*p, self.priority))
            .is_some_and(|(priority, _)| *priority < self.priority)
    }

    /// Let more urgent work go first, then queue again. Fails if the task is
    /// cancelled meanwhile (only for cancellable permits).
    pub fn yield_to_higher_priority(&mut self) -> Result<(), Cancelled> {
        if self.should_yield() {
            self.release();
            *self = self.lane.acquire(self.priority, self.cancel.clone())?;
        }
        Ok(())
    }

    fn release(&mut self) {
        if self.held {
            let mut state = self.lane.state.lock().unwrap();
            state.active[self.priority as usize] -= 1;
            self.held = false;
            self.lane.changed.notify_all();
        }
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    fn waiting(lane: &Lane, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut state = lane.state.lock().unwrap();
        while state.waiting.len() < count {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .expect("worker did not queue");
            state = lane.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }

    #[test]
    fn pending_gpu_work_uses_priority_then_fifo() {
        let scheduler = Scheduler::default();
        let hold = scheduler.gpu.acquire(Priority::Interactive, None).unwrap();
        let (tx, rx) = mpsc::channel();
        let mut workers = Vec::new();
        for (i, priority) in [
            Priority::Background,
            Priority::Export,
            Priority::Interactive,
            Priority::Interactive,
            Priority::Drag,
        ]
        .into_iter()
        .enumerate()
        {
            let lane = scheduler.gpu.clone();
            let tx = tx.clone();
            workers.push(thread::spawn(move || {
                let _permit = lane.acquire(priority, None).unwrap();
                tx.send(i).unwrap();
            }));
            waiting(&scheduler.gpu, i + 1);
        }
        drop(hold);
        for expected in [4, 2, 3, 1, 0] {
            assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), expected);
        }
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn background_gpu_work_yields_and_resumes() {
        let scheduler = Scheduler::default();
        let mut background = scheduler.gpu.acquire(Priority::Background, None).unwrap();
        let lane = scheduler.gpu.clone();
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _permit = lane.acquire(Priority::Interactive, None).unwrap();
            tx.send(()).unwrap();
        });
        waiting(&scheduler.gpu, 1);
        assert!(background.should_yield());
        background.yield_to_higher_priority().unwrap();
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(background.held);
        worker.join().unwrap();
    }

    #[test]
    fn cpu_reserves_foreground_capacity_and_cancellation_wakes_waiters() {
        let scheduler = Scheduler::default();
        let background = scheduler.task(Priority::Background);
        let _background = background.cpu().unwrap();
        let interactive = scheduler.task(Priority::Interactive);
        let _interactive = interactive.cpu().unwrap();
        let cancelled = scheduler.task(Priority::Background);
        let task = cancelled.clone();
        let worker = thread::spawn(move || task.cpu().is_err());
        waiting(&scheduler.cpu, 1);
        cancelled.cancel();
        assert!(worker.join().unwrap());
        assert!(scheduler.cpu.state.lock().unwrap().waiting.is_empty());
        let _gpu = scheduler.gpu.acquire(Priority::Interactive, None).unwrap();
    }

    #[test]
    fn panics_release_capacity_and_restore_the_thread_context() {
        let scheduler = Scheduler::default();
        let task = scheduler.task(Priority::Background);
        let result = std::panic::catch_unwind(|| {
            let _scope = task.enter();
            let _cpu = cpu_permit().unwrap();
            panic!("worker failed");
        });
        assert!(result.is_err());
        assert!(cpu_permit().unwrap().is_none());
        assert_eq!(scheduler.cpu.state.lock().unwrap().active, [0; 5]);
        let _cpu = task.cpu().unwrap();
    }
    #[test]
    fn exports_leave_room_for_visible_work_and_speculation_waits() {
        let scheduler = Scheduler::default();
        let _export = scheduler.task(Priority::Export).cpu().unwrap();
        // Thumbnails still run during a long export.
        let visible = scheduler.task(Priority::Visible).cpu().unwrap();
        drop(visible);
        let interactive = scheduler.task(Priority::Interactive).cpu().unwrap();
        let state = |lane: &Lane| {
            let state = lane.state.lock().unwrap();
            (state.waiting.len(), state.count(&[Priority::Background]))
        };
        // Speculative work waits for both the export and the foreground.
        let speculative = scheduler.task(Priority::Background);
        let task = speculative.clone();
        let worker = thread::spawn(move || task.cpu().map(drop).is_ok());
        waiting(&scheduler.cpu, 1);
        drop(_export);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(state(&scheduler.cpu), (1, 0), "waits for the foreground");
        drop(interactive);
        assert!(worker.join().unwrap());
    }

    #[test]
    fn refine_renders_yield_to_drag_frames_and_cancellation_stops_them() {
        let scheduler = Scheduler::default();
        let refine = scheduler.task(Priority::Interactive);
        let mut permit = {
            let _scope = refine.enter();
            gpu_permit_cancellable().unwrap().unwrap()
        };
        assert!(permit.is_preemptible());
        let lane = scheduler.gpu.clone();
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let drag = lane.acquire(Priority::Drag, None).unwrap();
            assert!(!drag.is_preemptible());
            tx.send(()).unwrap();
            // Hold the GPU until the refine has been cancelled.
            thread::sleep(Duration::from_millis(50));
        });
        waiting(&scheduler.gpu, 1);
        assert!(permit.should_yield());
        let yielded = thread::spawn(move || permit.yield_to_higher_priority().is_err());
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        refine.cancel();
        assert!(yielded.join().unwrap(), "cancelled while yielding");
        worker.join().unwrap();
        // A cancelled task is not admitted at all.
        let _scope = refine.enter();
        assert!(gpu_permit_cancellable().is_err());
        assert!(check().is_err());
    }

    #[test]
    fn background_tasks_run_on_the_background_pool() {
        let scheduler = Scheduler::default();
        let task = scheduler.task(Priority::Export);
        let name = task.run(|| {
            assert!(cpu_permit().unwrap().is_some(), "task is entered");
            std::thread::current().name().map(str::to_owned)
        });
        assert!(name.unwrap().starts_with("spektrafilm-background"));
        let foreground = scheduler.task(Priority::Interactive);
        let name = foreground.run(|| std::thread::current().name().map(str::to_owned));
        assert!(
            !name
                .unwrap_or_default()
                .starts_with("spektrafilm-background")
        );
    }
}
