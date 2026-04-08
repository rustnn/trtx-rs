use std::path::PathBuf;

use clap::Parser;
use clap_complete::Shell;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// ONNX files to compile and execute
    pub onnx: Vec<PathBuf>,

    /// Directory to store/cache engines
    #[arg(long, value_enum)]
    pub engine_dir: PathBuf,

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
    #[arg(long, short)]
    pub skip_inference: bool,
}
