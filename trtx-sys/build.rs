use std::env;
use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Check if we're in mock mode
    if env::var("CARGO_FEATURE_MOCK").is_ok() {
        println!("cargo:warning=Building in MOCK mode - no TensorRT-RTX required");

        // Build mock C implementation
        cc::Build::new().file("mock.c").compile("trtx_mock");

        generate_mock_bindings(&out_path);
        return;
    }

    println!("cargo:rerun-if-changed=wrapper.hpp");
    println!("cargo:rerun-if-changed=wrapper.cpp");

    // Look for TensorRT-RTX installation
    // Users can override with TENSORRT_RTX_DIR environment variable
    let trtx_dir =
        env::var("TENSORRT_RTX_DIR").unwrap_or_else(|_| "/usr/local/tensorrt-rtx".to_string());

    let include_dir = format!("{}/include", trtx_dir);
    let lib_dir = format!("{}/lib", trtx_dir);

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvonnxparser");

    // Also need CUDA runtime
    if let Ok(cuda_dir) = env::var("CUDA_ROOT") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_dir);
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        // Common CUDA locations
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    // Build C++ wrapper
    cc::Build::new()
        .cpp(true)
        .file("wrapper.cpp")
        .include(&include_dir)
        .flag("-std=c++17")
        .compile("trtx_wrapper");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.hpp")
        .clang_arg(format!("-I{}", include_dir))
        .allowlist_function("trtx_.*")
        .allowlist_type("TrtxLogger.*")
        .allowlist_var("TRTX_.*")
        .derive_debug(true)
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn generate_mock_bindings(out_path: &PathBuf) {
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
}
"#;

    std::fs::write(out_path.join("bindings.rs"), mock_bindings)
        .expect("Couldn't write mock bindings!");
}
