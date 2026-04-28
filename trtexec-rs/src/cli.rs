use std::path::PathBuf;

use clap::Parser;
use clap_complete::Shell;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// ONNX files or WebNN files to compile and execute
    pub inputs: Vec<PathBuf>,

    /// Directory to store/cache engines
    #[arg(long, value_enum)]
    pub engine_dir: PathBuf,

    /// Save engine to file path
    #[arg(long, short)]
    pub save_engine: Vec<PathBuf>,

    /// Generate shell completion to stdout
    #[arg(long, value_enum)]
    pub shell_completion: Option<Shell>,

    /// Number of inferences
    #[arg(long, short, default_value_t = 100)]
    pub num_inferences: u64,

    /// Use cuda graphs
    #[arg(long)]
    pub cuda_graph: bool,

    /// Set extra nvtx spans
    #[arg(long)]
    pub nvtx: bool,

    /// Don't use engines from cache
    #[arg(long)]
    pub no_cache: bool,

    /// Don't run inference
    #[arg(long)]
    pub skip_inference: bool,

    /// Which CUDA device index to use
    #[arg(long, default_value_t = 0)]
    pub cuda_device_idx: usize,

    /// Fail on dynamic input shapes instead of prompting (for scripts/CI)
    #[arg(long)]
    pub non_interactive: bool,

    #[cfg(unix)]
    #[arg(short, long)]
    /// Capture TensorRT API to JSON
    pub api_capture: Option<PathBuf>,

    /// Report layer time
    #[arg(long)]
    pub report_layer_time: bool,

    /// Write TensorRT layer timing profile JSON (trtexec-style); implies layer profiling
    #[arg(long, value_name = "PATH")]
    pub profile_json: Option<PathBuf>,
}
