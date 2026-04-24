use log::info;
use trtx::interfaces::ReportLayerTime;

/// Forwards TensorRT per-layer timings to the application log (see `IExecutionContext::setProfiler`).
/// To get a JSON like trtexec one would need to implement samples/common/sampleReporting.h in
/// particular     void exportJSONProfile(std::string const& fileName) const noexcept;
pub struct LayerTimingLogger;

impl ReportLayerTime for LayerTimingLogger {
    fn report_layer_time(&self, layer_name: &str, ms: f32) {
        info!("TensorRT layer {layer_name:?}: {ms:.4} ms");
    }
}
