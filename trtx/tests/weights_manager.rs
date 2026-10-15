#![cfg(all(
    feature = "v_1_7",
    feature = "dlopen_tensorrt_rtx",
    not(feature = "mock_runtime"),
    not(feature = "enterprise")
))]

use trtx::{Builder, BuilderFlag, DataType, Logger, Refitter, Runtime, TensorLocation};

#[test]
fn stripped_weights_survive_unload_and_restore() -> trtx::Result<()> {
    trtx::dynamically_load_tensorrt(None::<String>)?;
    let cuda = cudarc::driver::CudaContext::new(0).expect("CUDA context");
    let stream = cuda.new_stream().expect("CUDA stream");
    let logger = Logger::stderr()?;
    let initial: Vec<f32> = (0..4096).map(|i| i as f32 * 0.25).collect();
    let bytes = unsafe {
        std::slice::from_raw_parts(initial.as_ptr().cast(), std::mem::size_of_val(&*initial))
    };
    let mut builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    let constant = network.add_constant(&[4096], bytes, DataType::kFLOAT, None)?;
    let output = constant.output(&network, 0)?;
    output.set_name(&mut network, "output")?;
    network.mark_output(&output);
    let mut config = builder.create_config()?;
    config.set_flag(BuilderFlag::kREFIT);
    config.set_flag(BuilderFlag::kSTRIP_PLAN);
    let plan = builder.build_serialized_network(&mut network, &mut config)?;
    let mut runtime = Runtime::new(&logger)?;
    let mut engine = runtime.deserialize_cuda_engine(&plan)?;

    {
        let mut refitter = Refitter::new(&engine, &logger)?;
        let names = refitter.all_weights(64)?;
        assert!(!names.is_empty());
        for name in names {
            let prototype = refitter.weights_prototype(&name);
            assert_eq!(prototype.count, initial.len() as i64);
            assert_eq!(DataType::from(prototype.type_.clone()), DataType::kFLOAT);
            unsafe {
                refitter.set_named_weights_with_location(
                    &name,
                    trtx_sys::nvinfer1::Weights {
                        type_: DataType::kFLOAT.into(),
                        values: initial.as_ptr().cast(),
                        count: initial.len() as i64,
                    },
                    TensorLocation::kHOST.into(),
                )?;
            }
        }
        refitter.refit_cuda_engine()?;
    }

    let mut manager = engine
        .create_weights_manager()?
        .expect("stripped weights manager");
    let size = usize::try_from(manager.size()).expect("nonnegative weight size");
    assert!(size > 0);
    let mut backup = vec![0u8; size];
    let mut restored = vec![0u8; size];
    let raw_stream = stream.cu_stream().cast();

    // Reject invalid lengths and overflowing ranges before touching pointers or handles.
    unsafe {
        assert!(manager
            .save_to_buffer(
                std::ptr::null_mut(),
                size - 1,
                TensorLocation::kHOST,
                raw_stream
            )
            .is_err());
        assert!(manager
            .restore_from_buffer(
                std::ptr::null(),
                size - 1,
                TensorLocation::kHOST,
                raw_stream
            )
            .is_err());
        assert!(manager
            .restore_from_vmm_allocation(0, i64::MAX, 1, raw_stream)
            .is_err());
        manager.save_to_buffer(
            backup.as_mut_ptr().cast(),
            size,
            TensorLocation::kHOST,
            raw_stream,
        )?;
    }
    stream.synchronize().expect("save completes");
    unsafe {
        manager.unload(raw_stream)?;
        manager.restore_from_buffer(
            backup.as_ptr().cast(),
            size,
            TensorLocation::kHOST,
            raw_stream,
        )?;
    }
    stream.synchronize().expect("restore completes");
    unsafe {
        manager.save_to_buffer(
            restored.as_mut_ptr().cast(),
            size,
            TensorLocation::kHOST,
            raw_stream,
        )?;
    }
    stream.synchronize().expect("second save completes");
    assert_eq!(backup, restored);
    Ok(())
}
