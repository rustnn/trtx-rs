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
        let logger = Logger::stderr().unwrap();

        // not linking let's builder creation fail
        #[cfg(not(feature = "link_tensorrt_rtx"))]
        assert!(matches!(
            Builder::new(&logger),
            Err(trtx::Error::TrtRtxLibraryNotLoaded)
        ));

        // Loading the library fixes the error
        trtx::dynamically_load_tensorrt(None::<String>).unwrap();

        let logger = Logger::stderr().unwrap();
        let builder = Builder::new(&logger).unwrap();
        let mut network = builder.create_network(0).unwrap();

        // not linking let's builder creation fail
        #[cfg(not(feature = "link_tensorrt_onnxparser"))]
        assert!(matches!(
            OnnxParser::new(&mut network, &logger),
            Err(trtx::Error::TrtOnnxParserLibraryNotLoaded)
        ));

        // Loading the library fixes the error
        trtx::dynamically_load_tensorrt_onnxparser(None::<String>).unwrap();
        OnnxParser::new(&mut network, &logger).unwrap();
    }
}
