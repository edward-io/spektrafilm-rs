//! Cooperative admission for image work. CPU and GPU have separate lanes so
//! decoding can overlap rendering. GPU stages keep their buffers while yielding.
use std::{
    cell::RefCell,
    collections::BTreeSet,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Interactive,
    Export,
    Background,
}

#[derive(Clone)]
pub struct Scheduler {
    cpu: Arc<Lane>,
    gpu: Arc<Lane>,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self {
            cpu: Arc::new(Lane::new(2)),
            gpu: Arc::new(Lane::new(1)),
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
            .acquire(self.priority, Some(&self.cancelled))
    }

    pub fn enter(&self) -> Scope {
        Scope {
            previous: CURRENT.with(|current| current.replace(Some(self.clone()))),
            thread: PhantomData,
        }
    }
}

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

pub fn cpu_permit() -> Result<Option<Permit>, Cancelled> {
    CURRENT.with(|current| current.borrow().as_ref().map(Task::cpu).transpose())
}

/// GPU methods complete their calculation; callers discard cancelled results.
/// Cancellation is checked between CPU stages and before GPU processing starts.
pub fn gpu_permit() -> Option<Permit> {
    CURRENT
        .with(|current| current.borrow().clone())
        .map(|task| {
            task.scheduler
                .gpu
                .acquire(task.priority, None)
                .expect("uncancellable GPU admission")
        })
}

struct Lane {
    capacity: usize,
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Default)]
struct State {
    active: usize,
    interactive: usize,
    next: u64,
    waiting: BTreeSet<(Priority, u64)>,
}

impl Lane {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        }
    }

    fn wake(&self) {
        let _state = self.state.lock().unwrap();
        self.changed.notify_all();
    }

    fn acquire(
        self: &Arc<Self>,
        priority: Priority,
        cancel: Option<&AtomicBool>,
    ) -> Result<Permit, Cancelled> {
        let mut state = self.state.lock().unwrap();
        let ticket = (priority, state.next);
        state.next += 1;
        state.waiting.insert(ticket);
        self.changed.notify_all();
        loop {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                state.waiting.remove(&ticket);
                self.changed.notify_all();
                return Err(Cancelled);
            }
            // Background CPU work leaves a slot available for a newly selected photo.
            let capacity = if self.capacity > 1 && priority != Priority::Interactive {
                1
            } else {
                self.capacity
            };
            let foreground_active =
                self.capacity > 1 && priority != Priority::Interactive && state.interactive > 0;
            if state.active < capacity
                && !foreground_active
                && state.waiting.first() == Some(&ticket)
            {
                state.waiting.remove(&ticket);
                state.active += 1;
                state.interactive += usize::from(priority == Priority::Interactive);
                self.changed.notify_all();
                return Ok(Permit {
                    lane: self.clone(),
                    priority,
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
    held: bool,
}

impl Permit {
    pub fn is_interactive(&self) -> bool {
        self.priority == Priority::Interactive
    }

    pub fn should_yield(&self) -> bool {
        self.priority != Priority::Interactive
            && self
                .lane
                .state
                .lock()
                .unwrap()
                .waiting
                .first()
                .is_some_and(|(priority, _)| *priority < self.priority)
    }

    pub fn yield_to_higher_priority(&mut self) {
        if self.should_yield() {
            self.release();
            *self = self
                .lane
                .acquire(self.priority, None)
                .expect("uncancellable GPU admission");
        }
    }

    fn release(&mut self) {
        if self.held {
            let mut state = self.lane.state.lock().unwrap();
            state.active -= 1;
            state.interactive -= usize::from(self.priority == Priority::Interactive);
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
        for expected in [2, 3, 1, 0] {
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
        background.yield_to_higher_priority();
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
        assert_eq!(scheduler.cpu.state.lock().unwrap().active, 0);
        let _cpu = task.cpu().unwrap();
    }
}
