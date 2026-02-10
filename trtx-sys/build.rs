use std::env;
use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let link_trt = env::var("CARGO_FEATURE_LINK_TENSORRT_RTX").is_ok();
    let link_trt_onnxparser = env::var("CARGO_FEATURE_LINK_TENSORRT_ONNXPARSER").is_ok();
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LINK_TENSORRT_RTX");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LINK_TENSORRT_ONNXPARSER");

    // Check if we're in mock mode
    if env::var("CARGO_FEATURE_MOCK").is_ok() {
        println!("cargo:warning=Building in MOCK mode - no TensorRT-RTX required");

        // Build mock C implementation
        cc::Build::new().file("mock.c").compile("trtx_mock");

        generate_mock_bindings(&out_path);
        return;
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=logger_bridge.hpp");
    println!("cargo:rerun-if-changed=logger_bridge.cpp");
    println!("cargo:rerun-if-env-changed=TENSORRT_RTX_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");

    // Look for TensorRT-RTX installation
    // Users can override with TENSORRT_RTX_DIR environment variable
    let trtx_dir = match env::var("TENSORRT_RTX_DIR") {
        Ok(dir) => {
            println!("cargo:warning=Using TENSORRT_RTX_DIR={}", dir);
            dir
        }
        Err(_) => {
            println!(
                "cargo:warning=TENSORRT_RTX_DIR not set, using default: /usr/local/tensorrt-rtx"
            );
            "/usr/local/tensorrt-rtx".to_string()
        }
    };

    let include_dir = format!("{}/include", trtx_dir);
    let lib_dir = format!("{}/lib", trtx_dir);

    println!("cargo:rustc-link-search=native={}", lib_dir);
    // TensorRT 10.x uses versioned library names
    if cfg!(target_os = "windows") {
        if link_trt {
            println!("cargo:rustc-link-lib=dylib=tensorrt_rtx_1_3");
        }
        if link_trt_onnxparser {
            println!("cargo:rustc-link-lib=dylib=tensorrt_onnxparser_rtx_1_3");
        }
    } else {
        if link_trt {
            println!("cargo:rustc-link-lib=dylib=tensorrt_rtx");
        }
        if link_trt_onnxparser {
            println!("cargo:rustc-link-lib=dylib=tensorrt_onnxparser_rtx");
        }
    }

    let cuda_dir = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_ROOT"))
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:warning=CUDA_DIR={}", cuda_dir);

    // Also need CUDA runtime
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_dir);
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_dir);
    }
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Build logger bridge C++ wrapper
    let mut cc_build = cc::Build::new();
    cc_build
        .cpp(true)
        .file("logger_bridge.cpp")
        .include(&include_dir);

    // Also include CUDA headers
    cc_build.include(format!("{}/include", cuda_dir));
    if link_trt {
        cc_build.define("TRTX_LINK_TENSORRT_RTX", "1");
    }
    if link_trt_onnxparser {
        cc_build.define("TRTX_LINK_TENSORRT_ONNXPARSER", "1");
    }

    // Use correct C++17 flag based on compiler
    if cfg!(target_os = "windows") && cfg!(target_env = "msvc") {
        cc_build.flag("/std:c++17");
        cc_build.flag("/wd4100"); // Disable unused parameter warning on MSVC
        cc_build.flag("/wd4996"); // Disable deprecated declaration warning on MSVC
    } else {
        cc_build.flag("-std=c++17");
        cc_build.flag("-Wno-unused-parameter"); // Suppress unused parameter warnings
        cc_build.flag("-Wno-deprecated-declarations"); // Suppress deprecated warnings
    }

    cc_build.compile("trtx_logger_bridge");

    // Build autocxx bindings for main TensorRT API
    // Prepare CUDA include paths for autocxx clang parser
    let mut clang_args = vec![
        "-std=c++17".to_string(),
        "-Wno-unused-parameter".to_string(), // Suppress unused parameter warnings from TensorRT headers
        "-Wno-deprecated-declarations".to_string(), // Suppress deprecated warnings from TensorRT headers
    ];

    clang_args.push(format!("-I{}/include", cuda_dir));

    let clang_args_refs: Vec<&str> = clang_args.iter().map(|s| s.as_str()).collect();

    let mut autocxx_build = autocxx_build::Builder::new("src/lib.rs", [&include_dir])
        .extra_clang_args(&clang_args_refs)
        .build()
        .expect("Failed to build autocxx bindings");

    // Add CUDA include paths for C++ compilation phase as well
    autocxx_build.include(format!("{}/include", cuda_dir));

    // Set C++17 standard and suppress warnings
    if cfg!(target_os = "windows") && cfg!(target_env = "msvc") {
        autocxx_build.flag("/std:c++17");
        autocxx_build.flag("/wd4100"); // Disable unused parameter warning
        autocxx_build.flag("/wd4996"); // Disable deprecated declaration warning
    } else {
        autocxx_build.flag("-std=c++17");
        autocxx_build.flag("-Wno-unused-parameter"); // Suppress unused parameter warnings
        autocxx_build.flag("-Wno-deprecated-declarations"); // Suppress deprecated warnings
    }

    autocxx_build.compile("trtx_autocxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
}

fn generate_mock_bindings(out_path: &std::path::Path) {
    let mock_bindings = r#"
// Mock bindings for development without TensorRT-RTX

// Error codes
pub const TRTX_SUCCESS: i32 = 0;
pub const TRTX_ERROR_INVALID_ARGUMENT: i32 = 1;
pub const TRTX_ERROR_OUT_OF_MEMORY: i32 = 2;
pub const TRTX_ERROR_RUNTIME_ERROR: i32 = 3;
pub const TRTX_ERROR_CUDA_ERROR: i32 = 4;
pub const TRTX_ERROR_UNKNOWN: i32 = 99;

// Logger severity levels
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TrtxLoggerSeverity {
    TRTX_SEVERITY_INTERNAL_ERROR = 0,
    TRTX_SEVERITY_ERROR = 1,
    TRTX_SEVERITY_WARNING = 2,
    TRTX_SEVERITY_INFO = 3,
    TRTX_SEVERITY_VERBOSE = 4,
}

// Opaque types (just markers in mock mode)
#[repr(C)]
pub struct TrtxLogger {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct TrtxBuilder {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct TrtxBuilderConfig {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct TrtxNetworkDefinition {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct TrtxRuntime {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct TrtxCudaEngine {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct TrtxExecutionContext {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct TrtxOnnxParser {
    _unused: [u8; 0],
}

// Logger callback type
pub type TrtxLoggerCallback = ::std::option::Option<
    unsafe extern "C" fn(
        user_data: *mut ::std::os::raw::c_void,
        severity: TrtxLoggerSeverity,
        msg: *const ::std::os::raw::c_char,
    ),
>;

// Stub implementations that return success
extern "C" {
    pub fn trtx_logger_create(
        callback: TrtxLoggerCallback,
        user_data: *mut ::std::os::raw::c_void,
        out_logger: *mut *mut TrtxLogger,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_logger_destroy(logger: *mut TrtxLogger);

    pub fn trtx_builder_create(
        logger: *mut TrtxLogger,
        out_builder: *mut *mut TrtxBuilder,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_builder_destroy(builder: *mut TrtxBuilder);

    pub fn trtx_builder_create_network(
        builder: *mut TrtxBuilder,
        flags: u32,
        out_network: *mut *mut TrtxNetworkDefinition,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_builder_create_builder_config(
        builder: *mut TrtxBuilder,
        out_config: *mut *mut TrtxBuilderConfig,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_builder_build_serialized_network(
        builder: *mut TrtxBuilder,
        network: *mut TrtxNetworkDefinition,
        config: *mut TrtxBuilderConfig,
        out_data: *mut *mut ::std::os::raw::c_void,
        out_size: *mut usize,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_builder_config_destroy(config: *mut TrtxBuilderConfig);

    pub fn trtx_builder_config_set_memory_pool_limit(
        config: *mut TrtxBuilderConfig,
        pool_type: i32,
        pool_size: usize,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_network_destroy(network: *mut TrtxNetworkDefinition);

    pub fn trtx_runtime_create(
        logger: *mut TrtxLogger,
        out_runtime: *mut *mut TrtxRuntime,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_runtime_destroy(runtime: *mut TrtxRuntime);

    pub fn trtx_runtime_deserialize_cuda_engine(
        runtime: *mut TrtxRuntime,
        data: *const ::std::os::raw::c_void,
        size: usize,
        out_engine: *mut *mut TrtxCudaEngine,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_engine_destroy(engine: *mut TrtxCudaEngine);

    pub fn trtx_cuda_engine_create_execution_context(
        engine: *mut TrtxCudaEngine,
        out_context: *mut *mut TrtxExecutionContext,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_engine_get_tensor_name(
        engine: *mut TrtxCudaEngine,
        index: i32,
        out_name: *mut *const ::std::os::raw::c_char,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_engine_get_nb_io_tensors(
        engine: *mut TrtxCudaEngine,
        out_count: *mut i32,
    ) -> i32;

    pub fn trtx_execution_context_destroy(context: *mut TrtxExecutionContext);

    pub fn trtx_execution_context_set_tensor_address(
        context: *mut TrtxExecutionContext,
        tensor_name: *const ::std::os::raw::c_char,
        data: *mut ::std::os::raw::c_void,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_execution_context_enqueue_v3(
        context: *mut TrtxExecutionContext,
        cuda_stream: *mut ::std::os::raw::c_void,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_free_buffer(buffer: *mut ::std::os::raw::c_void);

    // ONNX Parser functions
    pub fn trtx_onnx_parser_create(
        network: *mut TrtxNetworkDefinition,
        logger: *mut TrtxLogger,
        out_parser: *mut *mut TrtxOnnxParser,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_onnx_parser_destroy(parser: *mut TrtxOnnxParser);

    pub fn trtx_onnx_parser_parse(
        parser: *mut TrtxOnnxParser,
        model_data: *const ::std::os::raw::c_void,
        model_size: usize,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    // CUDA Memory Management functions
    pub fn trtx_cuda_malloc(
        ptr: *mut *mut ::std::os::raw::c_void,
        size: usize,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_free(
        ptr: *mut ::std::os::raw::c_void,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_memcpy_host_to_device(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        size: usize,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_memcpy_device_to_host(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        size: usize,
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_synchronize(
        error_msg: *mut ::std::os::raw::c_char,
        error_msg_len: usize,
    ) -> i32;

    pub fn trtx_cuda_get_default_stream() -> *mut ::std::os::raw::c_void;
}

// Mock nvinfer1 module - stub types for trtx crate compatibility in mock mode
pub mod nvinfer1 {
    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum DataType {
        kFLOAT = 0,
        kHALF = 1,
        kINT8 = 2,
        kINT32 = 3,
        kBOOL = 4,
        kUINT8 = 5,
        kFP8 = 6,
        kBF16 = 7,
        kINT64 = 8,
        kINT4 = 9,
        kFP4 = 10,
        kE8M0 = 11,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ActivationType {
        kRELU = 0,
        kSIGMOID = 1,
        kTANH = 2,
        kLEAKY_RELU = 3,
        kELU = 4,
        kSELU = 5,
        kSOFTSIGN = 6,
        kSOFTPLUS = 7,
        kCLIP = 8,
        kHARD_SIGMOID = 9,
        kSCALED_TANH = 10,
        kTHRESHOLDED_RELU = 11,
        kGELU_ERF = 12,
        kGELU_TANH = 13,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum PoolingType {
        kMAX = 0,
        kAVERAGE = 1,
        kMAX_AVERAGE_BLEND = 2,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ElementWiseOperation {
        kSUM = 0,
        kPROD = 1,
        kMAX = 2,
        kMIN = 3,
        kSUB = 4,
        kDIV = 5,
        kPOW = 6,
        kFLOOR_DIV = 7,
        kAND = 8,
        kOR = 9,
        kXOR = 10,
        kEQUAL = 11,
        kGREATER = 12,
        kLESS = 13,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum MatrixOperation {
        kNONE = 0,
        kTRANSPOSE = 1,
        kVECTOR = 2,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum UnaryOperation {
        kEXP = 0,
        kLOG = 1,
        kSQRT = 2,
        kRECIP = 3,
        kABS = 4,
        kNEG = 5,
        kSIN = 6,
        kCOS = 7,
        kTAN = 8,
        kSINH = 9,
        kCOSH = 10,
        kASIN = 11,
        kACOS = 12,
        kATAN = 13,
        kASINH = 14,
        kACOSH = 15,
        kATANH = 16,
        kCEIL = 17,
        kFLOOR = 18,
        kERF = 19,
        kNOT = 20,
        kROUND = 21,
        kSIGN = 22,
        kISINF = 23,
        kISNAN = 24,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ReduceOperation {
        kSUM = 0,
        kPROD = 1,
        kMAX = 2,
        kMIN = 3,
        kAVG = 4,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum CumulativeOperation {
        kSUM = 0,
        kPROD = 1,
        kMIN = 2,
        kMAX = 3,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum GatherMode {
        kDEFAULT = 0,
        kELEMENT = 1,
        kND = 2,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ScatterMode {
        kELEMENT = 0,
        kND = 1,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum InterpolationMode {
        kNEAREST = 0,
        kLINEAR = 1,
        kCUBIC = 2,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ResizeCoordinateTransformation {
        kASYMMETRIC = 0,
        kALIGN_CORNERS = 1,
        kHALF_PIXEL = 2,
        kHALF_PIXEL_SYMMETRIC = 3,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ResizeRoundMode {
        kFLOOR = 0,
        kCEIL = 1,
        kROUND = 2,
        kHALF_UP = 3,
        kHALF_DOWN = 4,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ResizeSelector {
        kFORMULA = 0,
        kSIZES = 1,
        kUPPER = 2,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum TopKOperation {
        kMAX = 0,
        kMIN = 1,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ScaleMode {
        kUNIFORM = 0,
        kCHANNEL = 1,
        kELEMENTWISE = 2,
    }

    #[repr(i32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub enum ExecutionContextAllocationStrategy {
        kSTATIC = 0,
        kUSER_MANAGED = 1,
    }

    // Layer interface types (opaque stubs for mock - only used in type positions)
    #[repr(C)]
    pub struct IShuffleLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IActivationLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IResizeLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ITopKLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IGatherLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IScatterLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ISelectLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IMatrixMultiplyLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ISoftMaxLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IReduceLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ICumulativeLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IPoolingLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IConvolutionLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IDeconvolutionLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IQuantizeLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IDequantizeLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IConstantLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IConcatenationLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IScaleLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ISliceLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IUnaryLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IIdentityLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IPaddingLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ICastLayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ITensor { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ILayer { _unused: [u8; 0] }
    #[repr(C)]
    pub struct INetworkDefinition { _unused: [u8; 0] }
    #[repr(C)]
    pub struct ICudaEngine { _unused: [u8; 0] }
    #[repr(C)]
    pub struct IExecutionContext { _unused: [u8; 0] }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Weights {
        pub type_: DataType,
        pub values: *const ::std::ffi::c_void,
        pub count: i64,
    }

    impl Weights {
        pub fn new_float(values_ptr: *const ::std::ffi::c_void, count_val: i64) -> Self {
            Self { type_: DataType::kFLOAT, values: values_ptr, count: count_val }
        }
        pub fn new_with_type(
            data_type: DataType,
            values_ptr: *const ::std::ffi::c_void,
            count_val: i64,
        ) -> Self {
            Self { type_: data_type, values: values_ptr, count: count_val }
        }
    }
}

// Dims64/Dims - mock version
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dims64 {
    pub nbDims: i32,
    pub d: [i64; 8],
}

pub type Dims = Dims64;

impl Dims64 {
    pub fn from_slice(dims: &[i64]) -> Self {
        let mut d = [0i64; 8];
        let nb_dims = dims.len().min(8) as i32;
        d[..nb_dims as usize].copy_from_slice(&dims[..nb_dims as usize]);
        Self { nbDims: nb_dims, d }
    }
    pub fn new_2d(d0: i64, d1: i64) -> Self {
        Self { nbDims: 2, d: [d0, d1, 0, 0, 0, 0, 0, 0] }
    }
    pub fn new_3d(d0: i64, d1: i64, d2: i64) -> Self {
        Self { nbDims: 3, d: [d0, d1, d2, 0, 0, 0, 0, 0] }
    }
    pub fn new_4d(d0: i64, d1: i64, d2: i64, d3: i64) -> Self {
        Self { nbDims: 4, d: [d0, d1, d2, d3, 0, 0, 0, 0] }
    }
}

// ResizeMode is InterpolationMode in TensorRT
pub use nvinfer1::InterpolationMode as ResizeMode;
"#;

    std::fs::write(out_path.join("bindings.rs"), mock_bindings)
        .expect("Couldn't write mock bindings!");
}
