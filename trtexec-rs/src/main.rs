use anyhow::bail;
use anyhow::{anyhow, Context, Result};
use clap::CommandFactory;
use clap::Parser;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use log::{debug, info};
use rustnn::{load_graph_from_path, GraphConverter};
use std::ffi::{c_void, OsString};
use std::fs::File;
use std::io::{Read, Write};
use std::ops::Deref;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::prelude::*;
//use tracing_log::LogTracer;
use trtx::host_memory::HostMemory;
use trtx::{Builder, Logger, OnnxParser, ProfilingVerbosity};
use trtx::{LayerInformationFormat, Runtime};

use crate::cli::Args;
use crate::progress_monitor::ProgressMonitor;

mod cli;
mod progress_monitor;

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
        tracing_subscriber::registry()
            .with(nvtx::tracing::NvtxLayer::default())
            .init();
    } else {
        pretty_env_logger::init();
    }
    let span = tracing::info_span!("trtexec-rs");
    let _enter = span.enter();

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
    for onnx_path in args.inputs.iter() {
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
            let graph = load_graph_from_path(onnx_path)?;
            let converted = rustnn::converters::TrtxConverter.convert(&graph)?;
            engine_bytes.push(converted.data.into());

            continue;
        }
        debug!("Processing as ONNX file: {onnx_path:?}");

        let mut network = builder.create_network(0)?;
        let mut parser = OnnxParser::new(&mut network, &logger)?;
        parser.parse_from_file(&onnx_path.to_string_lossy().clone(), 5)?;

        for i in 0..network.get_nb_inputs() {
            let input = network.get_input(i)?;
            let shape = input.dimensions(&network)?;
            if shape.iter().any(|&i| i < 0) {
                let name = input.name(&network)?;
                bail!(
                    "Dynamic shapes are not supported. {name:?} requires dynamic shapes: {shape:?}"
                );
            }
        }

        let serialized = builder
            .build_serialized_network(&mut network, &mut config)
            .with_context(|| format!("Fail to build {onnx_path:?}"))?;

        std::fs::write(&cache_path, serialized.as_ref())
            .with_context(|| format!("Failed to write engine cache {cache_path:?}"))?;
        log::info!("Wrote engine cache {cache_path:?} ({hex_id})");
        println!("Wrote engine cache {cache_path:?} ({hex_id})");

        engine_bytes.push(serialized.into());
    }

    if args.skip_inference {
        return Ok(());
    }

    let cuda_ctx = CudaContext::new(args.cuda_device_idx)?;
    let mut runtime =
        Runtime::new(&logger).with_context(|| "Failed to create TensorRT runtime for inference")?;
    let stream = cuda_ctx.new_stream()?;

    for (bytes, input_path) in engine_bytes.drain(..).zip(args.inputs.iter()) {
        let mut engine = runtime.deserialize_cuda_engine(&bytes)?;
        let mut ctx = engine.create_execution_context()?;

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
            let name = engine.tensor_name(i)?;
            if engine.is_shape_inference_io(&name)? {
                bail!("Dynamic shapes are not supported. {name:?} requires dynamic shapes");
            }

            let dtype = engine.tensor_dtype(&name)?;
            let shape = engine.tensor_shape(&name)?;
            let num_elements = shape.iter().product::<i64>();
            // only correct for non-vectorized layouts
            let size = (num_elements as usize * dtype.size_bits()) / 8;
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
            "Duration CPU: {duration_cpu:?} ms ({} runs) {:?} ms per inference",
            args.num_inferences,
            duration_cpu / args.num_inferences as u32
        );
        println!(
            "Duration GPU: {duration_gpu:.02} ms ({} runs), {:?} ms per inference",
            args.num_inferences,
            duration_gpu / args.num_inferences as f32
        )
    }

    Ok(())
}
