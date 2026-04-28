use anyhow::bail;
use anyhow::{anyhow, Context, Result};
use clap::CommandFactory;
use clap::Parser;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use log::{debug, info};
use rustnn::load_graph_from_path;
use std::ffi::{c_void, OsString};
use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::ops::Deref;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{prelude::*, EnvFilter};
use trtx::host_memory::HostMemory;
use trtx::{Builder, Logger, OnnxParser, ProfilingVerbosity};
use trtx::{LayerInformationFormat, Runtime};

use crate::cli::Args;
use crate::profiler::{LayerProfilerReporter, LayerProfilerState, LayerTimingLogger};
use crate::progress_monitor::ProgressMonitor;

mod cli;
mod profiler;
mod progress_monitor;

/// If any dimension is negative (dynamic), either bail (`non_interactive`) or read concrete sizes from stdin.
fn resolve_dynamic_input_shape(
    non_interactive: bool,
    tensor_name: &str,
    shape: &[i64],
) -> Result<Vec<i64>> {
    if !shape.iter().any(|&d| d < 0) {
        return Ok(shape.to_vec());
    }
    if non_interactive {
        bail!(
            "Dynamic shapes are not supported in non-interactive mode. \
             {tensor_name:?} has shape {shape:?}; use an interactive terminal or set static shapes on the model."
        );
    }

    println!("Input tensor {tensor_name:?} has dynamic dimensions. Current shape: {shape:?}");
    println!("Enter a positive integer for each dynamic dimension (index in brackets):");
    let stdin = std::io::stdin();
    let mut reader = std::io::BufReader::new(stdin.lock());
    let mut resolved = shape.to_vec();

    for (idx, dim) in resolved.iter_mut().enumerate() {
        if *dim >= 0 {
            continue;
        }
        print!("  dimension[{idx}] (dynamic, was {dim}): ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let value: i64 = line.trim().parse().with_context(|| {
            format!("invalid integer for dimension[{idx}] of tensor {tensor_name:?}")
        })?;
        if value <= 0 {
            bail!("dimension[{idx}] of {tensor_name:?} must be positive, got {value}");
        }
        *dim = value;
    }

    Ok(resolved)
}

fn digest_hex(digest: &md5::Digest) -> String {
    format!("{digest:x}")
}

enum HostMemoryOrVec<'memory> {
    HostMemory(HostMemory<'memory>),
    Vec(Vec<u8>),
}

impl<'memory> AsRef<[u8]> for HostMemoryOrVec<'memory> {
    fn as_ref(&self) -> &[u8] {
        match self {
            HostMemoryOrVec::HostMemory(host_memory) => host_memory.as_ref(),
            HostMemoryOrVec::Vec(items) => items.as_ref(),
        }
    }
}

impl<'memory> Deref for HostMemoryOrVec<'memory> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'buffer> From<HostMemory<'buffer>> for HostMemoryOrVec<'buffer> {
    fn from(value: HostMemory<'buffer>) -> Self {
        HostMemoryOrVec::HostMemory(value)
    }
}
impl From<Vec<u8>> for HostMemoryOrVec<'_> {
    fn from(value: Vec<u8>) -> Self {
        HostMemoryOrVec::Vec(value)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(shell) = args.shell_completion {
        let mut cmd = Args::command();
        let bin_name = cmd.get_name().to_string();

        clap_complete::generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
        return Ok(());
    }

    if args.nvtx {
        // Send tracing/log events to nvtx
        let env_filter =
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::WARN.into())
                    .parse("")
                    .unwrap()
            });
        let fmt = tracing_subscriber::fmt::layer()
            .compact()
            .with_writer(std::io::stderr)
            .without_time()
            .with_filter(env_filter);
        tracing_subscriber::registry()
            .with(nvtx::tracing::NvtxLayer::default())
            .with(fmt)
            .init();
    } else {
        pretty_env_logger::init();
    }
    let span = tracing::info_span!("trtexec-rs");
    let _enter = span.enter();

    #[cfg(unix)]
    if let Some(json) = args.api_capture {
        // SAFETY: no threads launched yet, save to set env
        unsafe {
            std::env::set_var("TRT_SHIM_OUTPUT_JSON_FILE", json);
        }
        trtx::dynamically_load_tensorrt(Some("libtensorrt_shim.so"))
            .with_context(|| "Failed to load libtensorrt_shim.so for TensorRT API capture")?;
    }

    let abort = Arc::new(AtomicU32::new(0));
    let abort_clone = Arc::clone(&abort);

    ctrlc::set_handler(move || {
        abort_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    })?;

    let engine_dir = &args.engine_dir;
    std::fs::create_dir_all(engine_dir)
        .with_context(|| format!("Failed to create engine cache dir {engine_dir:?}"))?;

    let logger = Logger::log_crate()?;
    let mut builder = Builder::new(&logger)?;
    let mut config = builder.create_config()?;
    config
        .set_progress_monitor(Box::new(ProgressMonitor::new(Arc::clone(&abort))))
        .with_context(|| "Failed to set ProgressMonitor")?;
    config.set_profiling_verbosity(ProfilingVerbosity::kDETAILED);

    let mut engine_bytes = Vec::<HostMemoryOrVec>::new();
    for (i, onnx_path) in args.inputs.iter().enumerate() {
        info!("Processing {onnx_path:?}");
        let _span = tracing::info_span!(
            "Building engine",
            onnx_path = onnx_path.to_string_lossy().to_string()
        );
        let _enter = _span.enter();
        if abort.load(Ordering::Relaxed) > 0 {
            bail!("aborted");
        }

        let digest = md5::compute(
            std::fs::canonicalize(onnx_path)?
                .to_string_lossy()
                .as_bytes(),
        );
        let hex_id = digest_hex(&digest);
        let cache_path = engine_dir.join(format!("{hex_id}.engine"));

        if cache_path.is_file() && !args.no_cache {
            log::info!("Using cached engine {cache_path:?} for {onnx_path:?}");
            let mut buffer = Vec::new();
            File::open(cache_path)?.read_to_end(&mut buffer)?;
            engine_bytes.push(buffer.into());
            continue;
        }

        let mut _graph = None;
        let mut network = builder.create_network(0)?;
        let mut parser = OnnxParser::new(&mut network, &logger)?;
        let file_extension = onnx_path.extension();
        if file_extension == Some(&OsString::from("json"))
            || file_extension == Some(&OsString::from("webnn"))
        {
            info!("Processing as RustNN file: {onnx_path:?}");
            let _span = tracing::info_span!(
                "RustNN conversion",
                graph_path = onnx_path.to_string_lossy().to_string()
            );
            let _enter = _span.enter();
            _graph = Some(load_graph_from_path(onnx_path)?);
            rustnn::converters::TrtxConverter::build_network(
                _graph.as_ref().unwrap(),
                &mut network,
            )?;
        } else {
            debug!("Processing as ONNX file: {onnx_path:?}");

            parser.parse_from_file(&onnx_path.to_string_lossy().clone(), 5)?;
        }

        let mut shape_changed = false;
        for i in 0..network.get_nb_inputs() {
            let mut input = network.get_input(i)?;
            let shape = input.dimensions(&network)?;
            let name = input.name(&network)?;
            let resolved = resolve_dynamic_input_shape(args.non_interactive, &name, &shape)?;
            if resolved != shape {
                shape_changed = true;
                input.set_dimensions(&mut network, &resolved);
            }
        }

        let serialized = builder
            .build_serialized_network(&mut network, &mut config)
            .with_context(|| format!("Fail to build {onnx_path:?}"))?;

        if !shape_changed {
            std::fs::write(&cache_path, serialized.as_ref())
                .with_context(|| format!("Failed to write engine cache {cache_path:?}"))?;
            log::info!("Wrote engine cache {cache_path:?} ({hex_id})");
            println!("Wrote engine cache {cache_path:?} ({hex_id})");
        }
        if let Some(save_path) = args.save_engine.get(i) {
            println!("Saving engine to  {save_path:?}");
            std::fs::write(save_path, serialized.as_ref())
                .with_context(|| format!("Failed to write engine cache {cache_path:?}"))?;
        }

        engine_bytes.push(serialized.into());
    }

    if args.skip_inference {
        return Ok(());
    }

    let cuda_ctx = CudaContext::new(args.cuda_device_idx)?;
    let mut runtime =
        Runtime::new(&logger).with_context(|| "Failed to create TensorRT runtime for inference")?;
    let stream = cuda_ctx.new_stream()?;

    for (idx, (bytes, input_path)) in engine_bytes.drain(..).zip(args.inputs.iter()).enumerate() {
        let mut engine = runtime.deserialize_cuda_engine(&bytes)?;
        let mut ctx = engine.create_execution_context()?;

        let layer_profiler_state = if args.profile_json.is_some() {
            Some(Arc::new(LayerProfilerState::new(args.report_layer_time)))
        } else {
            None
        };

        if let Some(ref state) = layer_profiler_state {
            ctx.set_profiler(Box::new(LayerProfilerReporter::new(Arc::clone(state))))
                .with_context(|| "Failed to set TensorRT layer profiler")?;
        } else if args.report_layer_time {
            ctx.set_profiler(Box::new(LayerTimingLogger))
                .with_context(|| "Failed to set TensorRT layer profiler")?;
        }

        let inspector = engine.create_engine_inspector()?;
        let engine_layer_info_json =
            inspector.get_engine_information(LayerInformationFormat::kJSON)?;
        let layer_json_path = args.engine_dir.join(format!(
            "{}.graph.json",
            input_path
                .file_name()
                .ok_or_else(|| anyhow!("could not get filename fore {input_path:?}"))?
                .to_string_lossy()
        ));
        File::create(&layer_json_path)
            .with_context(|| {
                format!("Failed to create file for engine layer information at {layer_json_path:?}")
            })?
            .write_all(engine_layer_info_json.as_bytes())?;
        info!("Wrote {layer_json_path:?}");

        let mut io_tensors = Vec::<CudaSlice<u8>>::new();

        for i in 0..engine.nb_io_tensors()? {
            let name = engine.io_tensor_name(i)?;
            if engine.is_shape_inference_io(&name)? {
                bail!("Dynamic shapes are not supported. {name:?} requires dynamic shapes");
            }

            let data_type = engine.tensor_data_type(&name)?;
            let vec_dim = engine.tensor_vectorized_dim(&name)?;
            let comps = engine.tensor_components_per_element(&name)?;
            let shape = engine.tensor_shape(&name)?;
            let num_elements = shape
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if i == vec_dim as usize {
                        //v.next_multiple_of(comps as i64) // nightly feature
                        (v + (comps as i64) - 1) / comps as i64 * comps as i64
                    } else {
                        v
                    }
                })
                .product::<i64>();
            // only correct for non-vectorized layouts
            let size = (num_elements as usize * data_type.size_bits()) / 8;
            //let bytes = engine.tensor_components_per_element(&name)?;
            //let comp_per_element = engine.tensor_components_per_element(&name)?;
            let mut buffer = stream.alloc_zeros(size)?;
            let (ptr, _) = buffer.device_ptr_mut(&stream);
            unsafe { ctx.set_tensor_address(&name, ptr as *mut c_void)? };

            io_tensors.push(buffer);
        }
        let span = tracing::info_span!("Inference");
        let _enter = span.enter();

        let (duration_cpu, duration_gpu) = if args.cuda_graph {
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            for _ in 0..args.num_inferences {
                unsafe { ctx.enqueue_v3(stream.cu_stream() as *mut c_void)? };
                if abort.load(Ordering::Relaxed) > 0 {
                    bail!("aborted");
                }
            }
            let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH)?.ok_or_else(|| anyhow!("Failed to capture cuda graph"))?;

            let before_cpu = Instant::now();
            let before_gpu = stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| "Failed to record CUDA event before inference")?;
            graph.launch()?;
            let after_gpu = stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| "Failed to record CUDA event after inference")?;

            stream.synchronize()?;
            let after_cpu = Instant::now();
            let duration_cpu = after_cpu - before_cpu;
            let duration_gpu = before_gpu.elapsed_ms(&after_gpu)?;

            (duration_cpu, duration_gpu)
        } else {
            let before_cpu = Instant::now();
            stream.synchronize()?;
            let before_gpu = stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| "Failed to record CUDA event before inference")?;
            for _ in 0..args.num_inferences {
                unsafe { ctx.enqueue_v3(stream.cu_stream() as *mut c_void)? };
                if abort.load(Ordering::Relaxed) > 0 {
                    bail!("aborted");
                }
            }
            let after_gpu = stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| "Failed to record CUDA event after inference")?;
            stream.synchronize()?;
            let after_cpu = Instant::now();

            let duration_cpu = after_cpu - before_cpu;
            let duration_gpu = before_gpu.elapsed_ms(&after_gpu)?;

            (duration_cpu, duration_gpu)
        };

        println!(
            "Duration CPU: {duration_cpu:?} ({} runs) {:?} per inference",
            args.num_inferences,
            duration_cpu / args.num_inferences as u32
        );
        println!(
            "Duration GPU: {duration_gpu:.02} ms ({} runs), {:?} ms per inference",
            args.num_inferences,
            duration_gpu / args.num_inferences as f32
        );

        if let (Some(state), Some(json_base)) =
            (layer_profiler_state.as_ref(), args.profile_json.as_ref())
        {
            let export_path = if args.inputs.len() > 1 {
                let stem = json_base
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("layer_time");
                json_base.with_file_name(format!("{stem}_{idx}.json"))
            } else {
                json_base.clone()
            };
            state
                .export_json_profile(&export_path)
                .with_context(|| format!("export layer time JSON to {export_path:?}"))?;
            info!("Wrote layer timing profile {export_path:?}");
        }
    }

    Ok(())
}
