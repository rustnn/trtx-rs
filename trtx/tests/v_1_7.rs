#![cfg(all(
    feature = "v_1_7",
    feature = "dlopen_tensorrt_rtx",
    not(feature = "mock_runtime")
))]

use trtx::{Builder, DLAWorkspaceAllocationStrategy, DataType, Logger, Runtime};

#[test]
#[cfg(not(feature = "enterprise"))]
fn squeeze_profile_stream_and_deferred_unload() -> trtx::Result<()> {
    trtx::dynamically_load_tensorrt(None::<String>)?;
    let cuda = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = cuda.new_stream().unwrap();
    let logger = Logger::stderr()?;
    let mut builder = Builder::new(&logger)?;
    let mut profile = builder.creata_optimization_profile()?;
    assert!(profile.profile_stream().is_null());
    unsafe { profile.set_profile_stream(stream.cu_stream().cast()) };
    assert_eq!(profile.profile_stream(), stream.cu_stream().cast());
    unsafe { profile.set_profile_stream(std::ptr::null_mut()) };
    assert!(profile.profile_stream().is_null());

    let mut network = builder.create_network(0)?;
    let input = network.add_input("input", DataType::kFLOAT, &[1, 2, 1, 3])?;
    let squeezed = network.add_squeeze_all(&input)?.output(&network, 0)?;
    assert_eq!(squeezed.dimensions(&network)?, vec![2, 3]);
    squeezed.set_name(&mut network, "output")?;
    network.mark_output(&squeezed);
    let mut config = builder.create_config()?;
    let plan = builder.build_serialized_network(&mut network, &mut config)?;

    let mut runtime = Runtime::new(&logger)?;
    assert_eq!(
        runtime.dla_workspace_allocation_strategy(),
        DLAWorkspaceAllocationStrategy::kDEFAULT
    );
    unsafe {
        runtime.set_dla_workspace_allocation_strategy(DLAWorkspaceAllocationStrategy::kDEFAULT)?
    };
    runtime.set_deferred_weights_loading(true);
    let mut engine = runtime.deserialize_cuda_engine(&plan)?;
    assert!(!engine.weights_loaded());
    engine.load_weights(&plan)?;
    assert!(engine.weights_loaded());
    unsafe { engine.unload_weights()? };
    assert!(!engine.weights_loaded());
    assert!(unsafe { engine.unload_weights() }.is_err());
    engine.load_weights(&plan)?;
    assert!(engine.weights_loaded());
    Ok(())
}

#[test]
#[cfg(feature = "dlopen_tensorrt_onnxparser")]
fn parser_observer_records_outlive_parser() -> trtx::Result<()> {
    use std::{cell::RefCell, rc::Rc};
    use trtx::onnx_parser::RefitTransformKind;
    use trtx::OnnxParser;

    trtx::dynamically_load_tensorrt(None::<String>)?;
    trtx::dynamically_load_tensorrt_onnxparser(None::<String>)?;
    let records = Rc::new(RefCell::new(Vec::new()));
    let logger = Logger::stderr()?;
    let mut builder = Builder::new(&logger)?;
    let network = builder.create_network(0)?;
    let mut parser = OnnxParser::new(network, &logger)?;
    let captured = records.clone();
    parser.set_refit_observer(move |record| captured.borrow_mut().push(record))?;
    let model = include_bytes!("data/super-resolution-10.onnx");
    parser.parse(model)?;
    drop(parser);
    let records = records.borrow();
    assert!(!records.is_empty());
    assert!(records
        .iter()
        .all(|r| !r.trt_name.is_empty() && r.count > 0));
    assert!(records
        .iter()
        .any(|r| r.kind == RefitTransformKind::Identity && !r.source_onnx_names.is_empty()));
    Ok(())
}
