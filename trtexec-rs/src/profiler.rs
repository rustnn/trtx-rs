use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use log::info;
use serde_json::{json, Value};
use trtx::interfaces::ReportLayerTime;

/// Forwards TensorRT per-layer timings to the application log (see `IExecutionContext::setProfiler`).
pub struct LayerTimingLogger;

impl ReportLayerTime for LayerTimingLogger {
    fn report_layer_time(&self, layer_name: &str, ms: f32) {
        info!("TensorRT layer {layer_name:?}: {ms:.4} ms");
    }
}

/// Shared profiler samples; keep an [`Arc<LayerProfilerState>`] in `main` and pass a clone into
/// [`LayerProfilerReporter`] for `ExecutionContext::set_profiler`.
pub struct LayerProfilerState {
    log_each: bool,
    inner: Mutex<ProfilerSamples>,
}

struct ProfilerSamples {
    /// First-seen execution order (typical TensorRT callback order).
    layer_order: Vec<String>,
    samples: HashMap<String, Vec<f32>>,
}

impl LayerProfilerState {
    pub fn new(log_each: bool) -> Self {
        Self {
            log_each,
            inner: Mutex::new(ProfilerSamples {
                layer_order: Vec::new(),
                samples: HashMap::new(),
            }),
        }
    }

    fn record(&self, layer_name: &str, ms: f32) {
        if self.log_each {
            info!("TensorRT layer {layer_name:?}: {ms:.4} ms");
        }
        let mut g = self
            .inner
            .lock()
            .expect("LayerProfilerState mutex poisoned");
        if !g.samples.contains_key(layer_name) {
            g.layer_order.push(layer_name.to_string());
        }
        g.samples
            .entry(layer_name.to_string())
            .or_default()
            .push(ms);
    }

    /// Writes a JSON array like trtexec / sample `exportJSONProfile`: first element `{ "count": … }`,
    /// then one object per layer with total, average, median, and percentage of total time.
    ///
    /// Uses [`serde_json::to_writer`], which serializes a value to an `io::Write` (serde_json does not
    /// ship a symbol named `serialize_to_writer`).
    pub fn export_json_profile(&self, file_name: impl AsRef<Path>) -> Result<()> {
        let g = self
            .inner
            .lock()
            .expect("LayerProfilerState mutex poisoned");
        let updates_count = updates_count(&g);

        let mut rows: Vec<Value> = vec![json!({ "count": updates_count })];

        let total_time_ms: f64 = g
            .layer_order
            .iter()
            .filter_map(|name| {
                g.samples
                    .get(name)
                    .map(|s| s.iter().map(|&x| x as f64).sum::<f64>())
            })
            .sum();

        for name in &g.layer_order {
            let Some(samples) = g.samples.get(name) else {
                continue;
            };
            if samples.is_empty() {
                continue;
            }
            let time_ms: f64 = samples.iter().map(|&x| x as f64).sum();
            let n = samples.len() as f64;
            let average_ms = time_ms / n;
            let median_ms = median_ms(samples) as f64;
            let percentage = if total_time_ms > 0.0 {
                time_ms / total_time_ms * 100.0
            } else {
                0.0
            };
            rows.push(json!({
                "name": name,
                "timeMs": time_ms,
                "averageMs": average_ms,
                "medianMs": median_ms,
                "percentage": percentage,
            }));
        }

        let path = file_name.as_ref();
        let file = File::create(path).with_context(|| format!("create profiler JSON {path:?}"))?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer(&mut w, &rows).context("write profiler JSON")?;
        w.flush().context("flush profiler JSON")?;
        Ok(())
    }
}

fn updates_count(g: &ProfilerSamples) -> u64 {
    if g.layer_order.is_empty() {
        return 0;
    }
    g.layer_order
        .iter()
        .filter_map(|n| g.samples.get(n).map(|v| v.len() as u64))
        .min()
        .unwrap_or(0)
}

/// Concrete [`ReportLayerTime`] for `Box<dyn ReportLayerTime>`; holds a clone of the same
/// [`Arc`] the caller keeps for [`LayerProfilerState::export_json_profile`].
pub struct LayerProfilerReporter {
    state: Arc<LayerProfilerState>,
}

impl LayerProfilerReporter {
    pub fn new(state: Arc<LayerProfilerState>) -> Self {
        Self { state }
    }
}

impl ReportLayerTime for LayerProfilerReporter {
    fn report_layer_time(&self, layer_name: &str, ms: f32) {
        self.state.record(layer_name, ms);
    }
}

fn median_ms(samples: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = samples.to_vec();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}
