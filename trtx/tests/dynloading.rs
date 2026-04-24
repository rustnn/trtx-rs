#[cfg(test)]
#[cfg(not(feature = "mock"))]
#[cfg(feature = "dlopen_tensorrt_rtx")]
#[cfg(feature = "dlopen_tensorrt_onnxparser")]
mod tests {
    use trtx::{Builder, Logger, OnnxParser};

    // this needs to be in a single test in a separate test file to be isolated into a  dedicated
    // test binary
    #[test]
    fn dynloading() {
        // Loading the library fixes the error
        trtx::dynamically_load_tensorrt(None::<String>).unwrap();

        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let network = builder.create_network(0).unwrap();

        // Loading the library fixes the error
        trtx::dynamically_load_tensorrt_onnxparser(None::<String>).unwrap();
        OnnxParser::new(network, &logger).unwrap();
    }
}
