//! Runtime for deserializing and managing TensorRT engines.
//!
//! [`Runtime`] wraps [`trtx_sys::nvinfer1::IRuntime`] (C++ [`nvinfer1::IRuntime`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_runtime.html)).
//! [`ExecutionContext`] wraps [`trtx_sys::nvinfer1::IExecutionContext`] (C++ [`nvinfer1::IExecutionContext`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_execution_context.html)).

use std::ffi::CString;
use std::marker::PhantomData;
use std::pin::Pin;

use cxx::UniquePtr;
use trtx_sys::nvinfer1;

pub use crate::cuda_engine::CudaEngine;
pub use crate::engine_inspector::EngineInspector;
use crate::error::{Error, Result};
use crate::interfaces::{DebugListener, ProcessDebugTensor};
use crate::logger::Logger;

/// [`trtx_sys::nvinfer1::IExecutionContext`] — C++ [`nvinfer1::IExecutionContext`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_execution_context.html).
pub struct ExecutionContext<'a> {
    inner: UniquePtr<nvinfer1::IExecutionContext>,
    _engine: std::marker::PhantomData<&'a CudaEngine<'a>>,
    debug_listener: Option<Pin<Box<DebugListener>>>,
}

impl<'a> ExecutionContext<'a> {
    pub(crate) unsafe fn from_ptr(
        execution_context: *mut nvinfer1::IExecutionContext,
    ) -> Result<Self> {
        #[cfg(not(feature = "mock_runtime"))]
        if execution_context.is_null() {
            return Err(Error::Runtime(
                "Failed to create ExecutionContext".to_string(),
            ));
        }
        Ok(ExecutionContext {
            inner: UniquePtr::from_raw(execution_context),
            _engine: Default::default(),
            debug_listener: None,
        })
    }

    /// See [nvinfer1::IExecutionContext::setDebugListener].
    /// The Rust bindings only allow setting the debug listener once per execution context.
    pub fn set_debug_listener(&mut self, listener: Box<dyn ProcessDebugTensor>) -> Result<()> {
        let debug_listener = DebugListener::new(listener)?;
        if self.debug_listener.is_some() {
            panic!("Setting a debug listener more than once not supported at the moment");
        }
        self.debug_listener = Some(debug_listener);
        #[cfg(not(feature = "mock_runtime"))]
        {
            let success = unsafe {
                self.inner.pin_mut().setDebugListener(
                    self.debug_listener
                        .as_ref()
                        .expect("debug_listener can't be empty, we just set it")
                        .as_raw(),
                )
            };
            if !success {
                self.debug_listener = None;
                return Err(Error::Runtime("setDebugListener failed".to_string()));
            }
        }
        Ok(())
    }

    /// See [nvinfer1::IExecutionContext::setTensorDebugState].
    pub fn set_tensor_debug_state(&mut self, name: &str, flag: bool) -> Result<()> {
        let name = CString::new(name)?;
        if !unsafe {
            self.inner
                .pin_mut()
                .setTensorDebugState(name.as_ptr(), flag)
        } {
            Err(Error::FailedToSetProperty(
                crate::error::PropertySetAttempt::ExecutionContextTensorDebugState,
            ))
        } else {
            Ok(())
        }
    }

    /// See [nvinfer1::IExecutionContext::getDebugState].
    pub fn get_tensor_debug_state(&self, name: &str) -> Result<bool> {
        let name = CString::new(name)?;
        unsafe { Ok(self.inner.getDebugState(name.as_ptr())) }
    }

    /// See [nvinfer1::IExecutionContext::setAllTensorsDebugState].
    pub fn set_all_tensors_debug_state(&mut self, flag: bool) -> Result<()> {
        if !self.inner.pin_mut().setAllTensorsDebugState(flag) {
            Err(Error::FailedToSetProperty(
                crate::error::PropertySetAttempt::ExecutionContextTensorDebugState,
            ))
        } else {
            Ok(())
        }
    }
    /// See [nvinfer1::IExecutionContext::setUnfusedTensorsDebugState].
    pub fn set_unfused_tensors_debug_state(&mut self, flag: bool) -> Result<()> {
        if !self.inner.pin_mut().setUnfusedTensorsDebugState(flag) {
            Err(Error::FailedToSetProperty(
                crate::error::PropertySetAttempt::ExecutionContextTensorDebugState,
            ))
        } else {
            Ok(())
        }
    }
    /// See [nvinfer1::IExecutionContext::getUnfusedTensorsDebugState].
    pub fn get_unfused_tensor_debug_state(&self) -> bool {
        self.inner.getUnfusedTensorsDebugState()
    }

    /// Binds a tensor to a device memory address.
    ///
    /// # Safety
    /// `data` must point to valid CUDA memory with at least the tensor's size in bytes,
    /// and remain valid for the duration of inference.
    pub unsafe fn set_tensor_address(
        &mut self,
        name: &str,
        data: *mut std::ffi::c_void,
    ) -> Result<()> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }
            let name_cstr = std::ffi::CString::new(name)?;
            let success = self
                .inner
                .pin_mut()
                .setTensorAddress(name_cstr.as_ptr(), data as *mut _);
            if !success {
                return Err(Error::Runtime("Failed to set tensor address".to_string()));
            }
        }
        Ok(())
    }

    /// Enqueues inference on the given CUDA stream.
    ///
    /// # Safety
    /// `cuda_stream` must be a valid CUDA stream, and all tensor addresses must
    /// point to valid device memory.
    pub unsafe fn enqueue_v3(&mut self, cuda_stream: *mut std::ffi::c_void) -> Result<()> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }
            let success = self.inner.pin_mut().enqueueV3(cuda_stream as *mut _);
            if !success {
                return Err(Error::Runtime("Failed to enqueue inference".to_string()));
            }
        }
        Ok(())
    }
}

/// [`trtx_sys::nvinfer1::IRuntime`] — C++ [`nvinfer1::IRuntime`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_runtime.html).
pub struct Runtime<'logger> {
    inner: UniquePtr<nvinfer1::IRuntime>,
    _logger: PhantomData<&'logger Logger>,
}

impl<'runtime> Runtime<'runtime> {
    #[cfg(not(feature = "link_tensorrt_rtx"))]
    #[cfg(not(feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'runtime Logger) -> Result<Self> {
        Err(Error::TrtRtxLibraryNotLoaded)
    }

    #[cfg(any(feature = "link_tensorrt_rtx", feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'runtime Logger) -> Result<Self> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            let logger_ptr = logger.as_logger_ptr();
            let runtime_ptr = {
                #[cfg(feature = "link_tensorrt_rtx")]
                unsafe {
                    trtx_sys::create_infer_runtime(logger_ptr)
                }
                #[cfg(not(feature = "link_tensorrt_rtx"))]
                #[cfg(feature = "dlopen_tensorrt_rtx")]
                unsafe {
                    use libloading::Symbol;
                    use std::ffi::c_void;

                    use crate::TRTLIB;
                    if !TRTLIB.read()?.is_some() {
                        crate::dynamically_load_tensorrt(None::<String>)?;
                    }

                    let lock = TRTLIB.read()?;
                    let create_infer_runtime: Symbol<fn(*mut c_void, u32) -> *mut c_void> = lock
                        .as_ref()
                        .ok_or(Error::TrtRtxLibraryNotLoaded)?
                        .get(b"createInferRuntime_INTERNAL")?;
                    create_infer_runtime(logger_ptr, trtx_sys::get_tensorrt_version())
                }
            } as *mut nvinfer1::IRuntime;
            if runtime_ptr.is_null() {
                return Err(Error::Runtime("Failed to create runtime".to_string()));
            }
            Ok(Runtime {
                inner: unsafe { UniquePtr::from_raw(runtime_ptr) },
                _logger: Default::default(),
            })
        }
        #[cfg(feature = "mock_runtime")]
        Ok(Runtime {
            inner: UniquePtr::null(),
            _logger: Default::default(),
        })
    }

    pub fn deserialize_cuda_engine(&'_ mut self, data: &[u8]) -> Result<CudaEngine<'runtime>> {
        if cfg!(feature = "mock_runtime") {
            Ok(unsafe { CudaEngine::from_ptr(std::ptr::null_mut()) })
        } else {
            unsafe {
                let engine = self.inner.pin_mut().deserializeCudaEngine(
                    data.as_ref().as_ptr() as *const autocxx::c_void,
                    data.len(),
                );
                Ok(CudaEngine::from_ptr(engine.as_mut().ok_or_else(|| {
                    Error::Runtime("Failed to deserialize engine".to_string())
                })?))
            }
        }
    }
    //pub fn deserialize_cuda_engine_v2(
    //&'_ mut self,
    //stream_reader: &'runtime mut StreamReaderV2,
    //) -> Result<CudaEngine<'runtime>> {
    //if cfg!(feature = "mock_runtime") {
    //Ok(unsafe { CudaEngine::from_ptr(std::ptr::null_mut()) })
    //} else {
    //unsafe {
    //let engine = self
    //.inner
    //.pin_mut()
    //.deserializeCudaEngine1(stream_reader.pin_mut());
    //Ok(CudaEngine::from_ptr(engine.as_mut().ok_or_else(|| {
    //Error::Runtime("Failed to deserialize engine".to_string())
    //})?))
    //}
    //}
    //}
}

#[cfg(test)]
#[cfg(not(feature = "mock_runtime"))]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::builder::{Builder, MemoryPoolType};
    use crate::cuda::{synchronize, DeviceBuffer};
    use crate::interfaces::{ProcessDebugTensor, ProcessDebugTensorResult};
    use crate::logger::Logger;
    use crate::{DataType, ElementWiseOperation, Runtime};
    use trtx_sys::{Dims64, TensorLocation};

    /// Builds a network: input tensor_0 [1] -> +1 -> tensor_1 -> +1 -> tensor_2 -> +1 -> tensor_3 -> +1 -> tensor_4 (output).
    /// Each intermediate tensor is named and marked for debug.
    fn build_plus1_chain(logger: &Logger) -> crate::Result<(Vec<u8>, Vec<String>)> {
        let mut builder = Builder::new(logger)?;
        let mut network = builder.create_network(0)?;

        let one_bytes = 1.0f32.to_le_bytes();
        let mut tensor = network.add_input("tensor_0", DataType::kFLOAT, &[1])?;
        let mut debug_names = Vec::new();

        for i in 1..=4 {
            let one_layer =
                network.add_small_constant_copied(&[1], &one_bytes, DataType::kFLOAT)?;
            let one_t = one_layer.get_output(&network, 0)?;
            let mut sum_layer =
                network.add_elementwise(&tensor, &one_t, ElementWiseOperation::kSUM)?;
            sum_layer.set_name(&mut network, &format!("plus1_{}", i))?;
            tensor = sum_layer.get_output(&network, 0)?;
            let name = format!("tensor_{}", i);
            tensor.set_name(&mut network, &name)?;
            network.mark_tensor_debug(&tensor)?;
            assert!(network.is_debug_tensor(&tensor));
            debug_names.push(name);
        }
        network.mark_output(&tensor);

        let mut config = builder.create_config()?;
        config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, 1 << 20);
        //config.set_flag(trtx_sys::BuilderFlag::kDEBUG);
        let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
        Ok((engine_data.to_vec(), debug_names))
    }

    type ExpectedResults = Vec<(String, Vec<i64>)>;
    /// Debug listener that collects tensor names and shapes for verification.
    struct CollectingDebugListener {
        seen: Arc<Mutex<ExpectedResults>>,
    }

    impl ProcessDebugTensor for CollectingDebugListener {
        unsafe fn process_debug_tensor(
            &self,
            _addr: *const std::ffi::c_void,
            _location: TensorLocation,
            _type_: DataType,
            shape: &Dims64,
            name: Option<&str>,
            _stream: *mut std::ffi::c_void,
        ) -> ProcessDebugTensorResult {
            let dims: Vec<i64> = shape
                .d
                .iter()
                .take(shape.nbDims as usize)
                .copied()
                .collect();
            self.seen
                .lock()
                .unwrap()
                .push((name.unwrap().to_string(), dims));
            Ok(())
        }
    }

    /// Builds a small conv network: input [1,1,4,4] -> conv(1->4) -> conv(4->4) -> conv(4->4) -> output.
    /// Each conv output is named and marked for debug.
    fn build_conv_chain(logger: &Logger) -> crate::Result<(Vec<u8>, Vec<String>)> {
        // Declare kernel bytes before builder so their lifetime outlives 'network.
        // conv0: out=4, in=1, 3x3  conv1/2: out=4, in=4, 3x3
        let make_kernel = |out_ch: usize, in_ch: usize| -> Vec<u8> {
            std::iter::repeat_n(0.1f32, out_ch * in_ch * 3 * 3)
                .flat_map(|v| v.to_le_bytes())
                .collect()
        };
        let kernel_0 = make_kernel(4, 1);
        let kernel_1 = make_kernel(4, 4);
        let kernel_2 = make_kernel(4, 4);

        let mut builder = Builder::new(logger)?;
        let mut network = builder.create_network(0)?;

        // Input: [N=1, C=1, H=4, W=4] — TensorRT conv requires at least 4D
        let mut tensor = network.add_input("input", DataType::kFLOAT, &[1, 1, 4, 4])?;
        let mut debug_names = Vec::new();

        let conv_defs: [(i32, &Vec<u8>); 3] = [(4, &kernel_0), (4, &kernel_1), (4, &kernel_2)];
        for (i, &(out_ch, kbytes)) in conv_defs.iter().enumerate() {
            let weights = crate::ConvWeights {
                kernel_weights: kbytes,
                kernel_dtype: DataType::kFLOAT,
                bias_weights: None,
                bias_dtype: None,
            };
            let mut conv = network.add_convolution(&tensor, out_ch, &[3, 3], &weights)?;
            conv.set_padding(&mut network, &[1i64, 1i64]);
            let name = format!("conv_out_{}", i);
            conv.set_name(&mut network, &name)?;
            tensor = conv.get_output(&network, 0)?;
            tensor.set_name(&mut network, &name)?;
            network.mark_tensor_debug(&tensor)?;
            debug_names.push(name);
        }
        network.mark_output(&tensor);

        let mut config = builder.create_config()?;
        config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, 1 << 20);
        let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
        Ok((engine_data.to_vec(), debug_names))
    }

    #[test]
    #[ignore = "only works on TRT enterprise at the moment"]
    fn set_debug_listener_conv_chain() {
        let logger = Logger::stderr().expect("logger");
        let (engine_data, _debug_names) = build_conv_chain(&logger).expect("build conv network");

        let mut runtime = Runtime::new(&logger).expect("runtime");
        let mut engine = runtime
            .deserialize_cuda_engine(&engine_data)
            .expect("deserialize");
        let mut context = engine
            .create_execution_context()
            .expect("execution context");

        let seen = Arc::new(Mutex::new(Vec::<(String, Vec<i64>)>::new()));
        context
            .set_debug_listener(Box::new(CollectingDebugListener {
                seen: Arc::clone(&seen),
            }))
            .expect("set_debug_listener");
        context.set_all_tensors_debug_state(true).unwrap();
        context.set_unfused_tensors_debug_state(true).unwrap();

        // input: 1 channel 4x4, output: 4 channels 4x4
        let input_elems = 4 * 4;
        let output_elems = 4 * 4 * 4;
        let elem_size = std::mem::size_of::<f32>();
        let input_bytes: Vec<u8> = std::iter::repeat_n(1.0f32, input_elems)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut input_device = DeviceBuffer::new(input_elems * elem_size).expect("input buffer");
        let output_device = DeviceBuffer::new(output_elems * elem_size).expect("output buffer");
        input_device
            .copy_from_host(&input_bytes)
            .expect("copy input");

        unsafe {
            context
                .set_tensor_address("input", input_device.as_ptr())
                .expect("set input");
            context
                .set_tensor_address("conv_out_2", output_device.as_ptr())
                .expect("set output");
            context
                .enqueue_v3(crate::cuda::get_default_stream())
                .expect("enqueue");
        }
        synchronize().expect("sync");

        let seen = seen.lock().unwrap();
        assert!(
            !seen.is_empty(),
            "debug listener should have seen at least one tensor, saw 0"
        );
    }

    #[test]
    #[ignore = "only works on TRT enterprise at the moment"]
    fn set_debug_listener_plus1_chain() {
        let logger = Logger::stderr().expect("logger");
        let (engine_data, expected_debug_names) =
            build_plus1_chain(&logger).expect("build network");
        assert_eq!(
            expected_debug_names,
            ["tensor_1", "tensor_2", "tensor_3", "tensor_4"]
        );

        let mut runtime = Runtime::new(&logger).expect("runtime");
        let mut engine = runtime
            .deserialize_cuda_engine(&engine_data)
            .expect("deserialize");
        let mut context = engine
            .create_execution_context()
            .expect("execution context");

        let seen = Arc::new(Mutex::new(Vec::<(String, Vec<i64>)>::new()));
        context
            .set_debug_listener(Box::new(CollectingDebugListener {
                seen: Arc::clone(&seen),
            }))
            .expect("set_debug_listener");
        context.set_all_tensors_debug_state(true).unwrap();
        context.set_unfused_tensors_debug_state(true).unwrap();

        let elem_size = std::mem::size_of::<f32>();
        let mut input_device = DeviceBuffer::new(elem_size).expect("input buffer");
        let output_device = DeviceBuffer::new(elem_size).expect("output buffer");
        input_device
            .copy_from_host(&0.0f32.to_le_bytes())
            .expect("copy input");

        unsafe {
            context
                .set_tensor_address("tensor_0", input_device.as_ptr())
                .expect("set input");
            context
                .set_tensor_address("tensor_4", output_device.as_ptr())
                .expect("set output");
            context
                .enqueue_v3(crate::cuda::get_default_stream())
                .expect("enqueue");
        }
        synchronize().expect("sync");

        let mut output_bytes = [0u8; 4];
        output_device
            .copy_to_host(&mut output_bytes)
            .expect("copy output");
        let output_val = f32::from_le_bytes(output_bytes);
        assert!(
            (output_val - 4.0f32).abs() < 1e-5,
            "expected output 4.0 (0+1+1+1+1), got {}",
            output_val
        );

        let seen = seen.lock().unwrap();
        assert!(
            seen.len() >= 4,
            "debug listener should see at least 4 tensors, saw {}",
            seen.len()
        );
        for expected in &expected_debug_names {
            assert!(
                seen.iter().any(|(n, _)| n.contains(expected.as_str())),
                "expected debug tensor {:?} among names {:?}",
                expected,
                seen.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
            );
        }
    }
}
