use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use bindgen::EnumVariation;
use regex::Regex;

/// Writes `contents` only if the file is missing or differs. Avoids bumping mtimes when the
/// build script re-runs unchanged; autocxx emits `cargo:rerun-if-changed` for headers under
/// `OUT_DIR`, and rewriting them every run made Cargo think those outputs changed after the
/// previous fingerprint (perpetual rebuild loop).
fn write_if_changed(path: &Path, contents: &[u8]) {
    let unchanged = std::fs::read(path)
        .map(|existing| existing == contents)
        .unwrap_or(false);
    if !unchanged {
        let mut file = File::create(path).unwrap();
        file.write_all(contents).unwrap();
    }
}

/// Creates a folder in `out_dir`, copies headers from `trt_dir` with transformations applied,
/// and returns the path to the new folder. Original headers are never modified.
fn prepare_transformed_headers(header_dir: &Path, out_dir: &Path) -> PathBuf {
    let doxy_regex = Regex::new(r"\\(\w+)").unwrap();
    let warn_regex = Regex::new(r"\\warning (.*)$").unwrap();
    let see_regex = Regex::new(r"\\see ([`\w:()]+)").unwrap();
    let param_regex = Regex::new(r"\\param ([\w:()]+)").unwrap();
    let comment_indent_regex = Regex::new(r"(///\ )(\ +)").unwrap();
    let http_link_regex = Regex::new(r"https:\/\/[\w.\/\-#]+").unwrap();

    let transformed_dir = out_dir.join("trtx_transformed_headers");
    std::fs::create_dir_all(&transformed_dir).expect("Failed to create transformed headers dir");

    for entry in std::fs::read_dir(header_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let replaced = std::fs::read_to_string(&path).unwrap();
            let replaced = warn_regex.replace_all(&replaced, "<div class=\"warning\"> $1 </div>");
            let replaced = see_regex.replace_all(&replaced, "See [`$1`]");
            let replaced = param_regex.replace_all(&replaced, "- `$1`");
            let replaced = http_link_regex.replace_all(&replaced, "<$0>");
            let replaced = replaced
                .replace("std::size_t", "size_t")
                // workaround autocxx limitation where there can't be the same type in different
                // namespaces
                .replace("namespace v_1_0", "inline namespace v_1_0")
                .replace("namespace impl", "inline namespace impl")
                .replace("ErrorCode getErrorCode", "int32_t getErrorCode")
                .replace(
                    "bool reportError(ErrorCode val",
                    "bool reportError(int32_t val",
                )
                .replace(
                    "void log(Severity severity, AsciiChar const* msg)",
                    "void log(int32_t severity, char const* msg)",
                )
                .replace("//!", "///")
                .replace(r"\returns", " - Returns ");
            let replaced = doxy_regex.replace_all(&replaced, "");
            // We need to declare everything added after v_1_3 because we can't do features for
            // autocxx
            let replaced = "#include <cstdint>\n".to_string()
                + &replaced
                + r#"
namespace nvinfer1 {
    class IMoELayer;
    class IDistCollectiveLayer;
    enum class ComputeCapability : int32_t;
}"#;

            // trimming of indentation is necessary, so that rustdoc doesn't interpret
            // indented sections as Rust code.
            let replaced = comment_indent_regex
                .replace_all(&replaced, "$1")
                .replace("\r\n", "\n");

            let out_file = transformed_dir.join(path.file_name().unwrap());
            write_if_changed(&out_file, replaced.as_bytes());
        }
    }

    transformed_dir
}

/// Generate enum bindings from NvInfer.h using bindgen (replaces generate_debug_enum.sh).
fn generate_enum_bindings(crate_root: &str, out_path: &Path, include_dir: &Path) {
    let header = include_dir.join("NvInfer.h").to_string_lossy().to_string();
    let cuda_shim = format!("{crate_root}/TensorRT-Headers");

    println!("cargo:rerun-if-changed={header}");

    let mut builder = bindgen::Builder::default()
        .header(header)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .blocklist_type("cu.*")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg(format!("-I{}", include_dir.to_string_lossy()))
        .clang_arg(format!("-I{cuda_shim}"));

    // Allowlist types matching the script's regex: (.*Type|.*Mode|.*Operation|...)
    for pattern in [
        ".*Type",
        ".*Mode",
        ".*Operation",
        ".*Strategy",
        ".*Severity",
        ".*Format",
        ".*Verbosity",
        ".*Feature",
        ".*Platform",
        ".*Level",
        ".*Capability",
        ".*ErrorCode",
        ".*Flag",
        ".*Selector",
        ".*Transformation",
        ".*Location",
        ".*Role",
        ".*Limit",
        ".*AttentionNormalizationOp",
        ".*SeekPosition",
    ] {
        builder = builder.allowlist_type(pattern);
    }
    builder = builder.blocklist_type(".*IPluginCapability");
    builder = builder.blocklist_type(".*IVersionedInterface");
    builder = builder.blocklist_type(".*InterfaceInfo");
    builder = builder.blocklist_type(".*InterfaceKind");

    let bindings = builder
        .generate()
        .expect("Failed to generate enum bindings from NvInfer.h");

    let mut output = bindings.to_string();
    output = output.replace("extern \"C\"", "extern \"system\"");
    output = output.replace("nvinfer1_", "");
    output = output.replace("ILogger_", "");
    output = output.replace("impl__EnumMaxImpl", "impl_EnumMaxImpl");

    let out_file = out_path.join("enums.rs");
    let enums_src = format!("/* automatically generated by bindgen */\n\n{output}");
    write_if_changed(&out_file, enums_src.as_bytes());
}

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let crate_root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let link_trt = env::var("CARGO_FEATURE_LINK_TENSORRT_RTX").is_ok();
    let link_trt_onnxparser = env::var("CARGO_FEATURE_LINK_TENSORRT_ONNXPARSER").is_ok();

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LINK_TENSORRT_RTX");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LINK_TENSORRT_ONNXPARSER");

    let trt_version = if cfg!(feature = "v_1_4") {
        "1.4"
    } else if cfg!(feature = "v_1_3") {
        "1.3"
    } else {
        panic!("No version feature enabled! Need to at least enable v_1_3 or v_1_4");
    };
    let trt_version_suffix = if cfg!(unix) || cfg!(feature = "enterprise") {
        ""
    } else {
        if cfg!(feature = "v_1_4") {
            "_1_4"
        } else {
            "_1_3"
        }
    };

    let header_overwrite = env::var("TENSORRT_INCLUDE_DIR").ok();
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    let lib_overwrite = env::var("TENSORRT_LIB_DIR").ok();
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    let sdk_overwrite = env::var("TENSORRT_SDK_DIR").ok();
    println!("cargo:rerun-if-env-changed=TENSORRT_SDK_DIR");
    if let Some(sdk_overwrite) = sdk_overwrite.as_ref() {
        println!("cargo:warning=Using TENSORRT_SDK_DIR={sdk_overwrite}");
    }
    if let Some(lib_overwrite) = lib_overwrite.as_ref() {
        println!("cargo:warning=Using TENSORRT_LIB_DIR={lib_overwrite}");
    }
    if let Some(header_overwrite) = header_overwrite.as_ref() {
        println!("cargo:warning=Using TENSORRT_INCLUDE_DIR={header_overwrite}");
    }

    let include_dir = header_overwrite
        .or(sdk_overwrite.clone().map(|p| format!("{p}/include")))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(if cfg!(feature = "enterprise") {
                format!("{crate_root}/TensorRT-Headers/TRT-Enterprise-10.16")
            } else {
                format!("{crate_root}/TensorRT-Headers/TRT-RTX-{trt_version}")
            })
        });
    println!("cargo:rerun-if-changed={}", include_dir.display());
    let cuda_shim_include_dir = format!("{crate_root}/TensorRT-Headers");
    let lib_dir = lib_overwrite.or(sdk_overwrite.map(|p| format!("{p}/lib")));

    // Generate enum bindings from NvInfer.h (used in both mock and real builds)
    generate_enum_bindings(&crate_root, &out_path, &include_dir);

    // Check if we're in mock mode
    let is_mock = env::var("CARGO_FEATURE_MOCK").is_ok();

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=logger_bridge.hpp");
    println!("cargo:rerun-if-changed=logger_bridge.cpp");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");

    let transformed_include_dir = prepare_transformed_headers(&include_dir, &out_path);
    let transformed_include_dir_str = transformed_include_dir.to_string_lossy();

    if let Some(lib_dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }
    if link_trt {
        if cfg!(feature = "enterprise") {
            println!("cargo:rustc-link-lib=dylib=nvinfer");
        } else {
            println!("cargo:rustc-link-lib=dylib=tensorrt_rtx{trt_version_suffix}");
        }
    }
    if link_trt_onnxparser {
        if cfg!(feature = "enterprise") {
            println!("cargo:rustc-link-lib=dylib=nvonnxparser");
        } else {
            println!("cargo:rustc-link-lib=dylib=tensorrt_onnxparser_rtx{trt_version_suffix}");
        }
    }

    // Build logger bridge C++ wrapper
    let mut cc_build = cc::Build::new();
    cc_build
        .cpp(true)
        .file("logger_bridge.cpp")
        .include(&transformed_include_dir)
        .include(&cuda_shim_include_dir);

    if is_mock {
        cc_build.define("TRTX_MOCK_MODE", "1");
    }
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
    let clang_args = vec![
        "-std=c++17",
        "-Wno-unused-parameter", // Suppress unused parameter warnings from TensorRT headers
        "-Wno-deprecated-declarations", // Suppress deprecated warnings from TensorRT headers
    ];

    let mut autocxx_build = autocxx_build::Builder::new(
        "src/lib.rs",
        [
            transformed_include_dir_str.as_ref(),
            cuda_shim_include_dir.as_str(),
        ],
    )
    .extra_clang_args(&clang_args)
    .build()
    .expect("Failed to build autocxx bindings");

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
}
