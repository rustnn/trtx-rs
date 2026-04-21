use std::{
    ops::ControlFlow,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use indicatif::ProgressBar;
use trtx::interfaces::MonitorProgress;

pub struct ProgressMonitor {
    total: AtomicU64,
    phases: AtomicU64,
    cancel_flag: Arc<AtomicU32>,
    pb: Mutex<ProgressBar>,
}

impl ProgressMonitor {
    pub fn new(cancel_flag: Arc<AtomicU32>) -> Self {
        Self {
            total: AtomicU64::new(0),
            phases: AtomicU64::new(0),
            cancel_flag,
            pb: ProgressBar::new(0).into(),
        }
    }
}

const CLONING_PHASE: &str = "Cloning network graph";

impl MonitorProgress for ProgressMonitor {
    fn phase_start(&self, phase_name: &str, _parent_phase: Option<&str>, num_steps: i32) {
        if phase_name == CLONING_PHASE {
            return;
        }
        self.total.fetch_add(num_steps as u64, Ordering::Relaxed);
        self.phases.fetch_add(1, Ordering::Relaxed);
        self.pb.lock().unwrap().inc_length(num_steps as u64);
    }

    fn step_complete(&self, phase_name: &str, _step: i32) -> ControlFlow<()> {
        if phase_name != CLONING_PHASE {
            self.pb.lock().unwrap().inc(1);
        }
        if self.cancel_flag.load(Ordering::Relaxed) > 0 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn phase_finish(&self, phase_name: &str) {
        if phase_name == CLONING_PHASE {
            return;
        }
        let total = self.phases.fetch_sub(1, Ordering::Relaxed) - 1;
        if total == 0 {
            self.pb.lock().unwrap().finish();
        }
    }
}
