//! Miette-based diagnostic representations for network errors.
//!
//! Provides [`WithNetworkContext`] which enriches a [`crate::Error`] with an annotated
//! text dump of the [`NetworkDefinition`] so that miette can highlight the
//! problematic layer or tensor and label its inputs.
//!
//! Enabled with the `miette` crate feature.

use std::{collections::HashMap, fmt};

use miette::{Diagnostic, LabeledSpan, NamedSource, SourceSpan};
use trtx_sys::AsLayer;

use crate::{
    error::Error,
    network::{Layer, NetworkDefinition},
    tensor::Tensor,
};

// ---- text formatting helpers -----------------------------------------------

fn fmt_dtype(dt: trtx_sys::DataType) -> String {
    match dt {
        trtx_sys::DataType::kFLOAT => "F32".to_string(),
        trtx_sys::DataType::kHALF => "F16".to_string(),
        trtx_sys::DataType::kBF16 => "BF16".to_string(),
        trtx_sys::DataType::kINT8 => "I8".to_string(),
        trtx_sys::DataType::kINT32 => "I32".to_string(),
        trtx_sys::DataType::kINT64 => "I64".to_string(),
        trtx_sys::DataType::kUINT8 => "U8".to_string(),
        trtx_sys::DataType::kBOOL => "Bool".to_string(),
        trtx_sys::DataType::kFP8 => "FP8".to_string(),
        trtx_sys::DataType::kFP4 => "FP4".to_string(),
        trtx_sys::DataType::kINT4 => "I4".to_string(),
        trtx_sys::DataType::kE8M0 => "E8M0".to_string(),
    }
}

fn fmt_dims(dims: &[i64]) -> String {
    let parts: Vec<String> = dims
        .iter()
        .map(|&d| {
            if d < 0 {
                "?".to_string()
            } else {
                d.to_string()
            }
        })
        .collect();
    format!("[{}]", parts.join(","))
}

fn fmt_layer_type(lt: trtx_sys::LayerType) -> String {
    let s = format!("{:?}", lt);
    let s = s.strip_prefix('k').unwrap_or(&s);
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
    }
}

fn fmt_tensor(network: &NetworkDefinition, tensor: Tensor) -> String {
    let name = tensor.name(network).unwrap_or_else(|_| "?".to_string());
    let dtype = fmt_dtype(tensor.data_type(network));
    let shape = tensor
        .dimensions(network)
        .map(|d| fmt_dims(&d))
        .unwrap_or_else(|_| "[?]".to_string());
    format!("{}({}, {})", name, shape, dtype)
}

fn make_span(start: usize, end: usize) -> SourceSpan {
    SourceSpan::new(start.into(), end - start)
}

// ---- network text builder --------------------------------------------------

struct NetworkText {
    source: String,
    /// Full-line span for each layer, keyed by raw ILayer pointer.
    layer_spans: HashMap<usize, SourceSpan>,
    /// Raw pointers of each layer's input tensors (for looking up their definition spans).
    layer_input_tensor_ptrs: HashMap<usize, Vec<usize>>,
    /// Names of each input tensor for a layer (parallel to `layer_input_tensor_ptrs`).
    layer_input_names: HashMap<usize, Vec<String>>,
    /// Where each tensor is *defined*: its span in the output section of its producing layer,
    /// or in the network-input block at the top.
    tensor_output_spans: HashMap<usize, SourceSpan>,
    /// For each tensor, which layer produced it (absent for network inputs).
    tensor_producers: HashMap<usize, usize>,
    /// Human-readable "LayerType[name]" string for each layer pointer.
    layer_display: HashMap<usize, String>,
}

/// Build a multi-line text representation of the network in the form:
///
/// ```text
/// input0([1,3,224,224], F32)  // network input
///
/// out([1,64,112,112], F32) <- Convolution[conv1](input0([1,3,224,224], F32), weight([64,3,3,3], F32))
/// ```
fn build_network_text(network: &NetworkDefinition) -> NetworkText {
    let net_output_ptrs: std::collections::HashSet<usize> = (0..network.nb_outputs())
        .filter_map(|i| network.output(i).ok())
        .map(|t| t.inner as usize)
        .collect();

    let mut source = String::new();
    let mut layer_spans: HashMap<usize, SourceSpan> = HashMap::new();
    let mut layer_input_tensor_ptrs: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut layer_input_names: HashMap<usize, Vec<String>> = HashMap::new();
    let mut tensor_output_spans: HashMap<usize, SourceSpan> = HashMap::new();
    let mut tensor_producers: HashMap<usize, usize> = HashMap::new();
    let mut layer_display: HashMap<usize, String> = HashMap::new();

    // Network input tensors
    for i in 0..network.nb_inputs() {
        let Ok(tensor) = network.input(i) else {
            continue;
        };
        let start = source.len();
        source.push_str(&fmt_tensor(network, tensor));
        let end = source.len();
        source.push_str("  // network input\n");
        tensor_output_spans.insert(tensor.inner as usize, make_span(start, end));
    }

    if network.nb_inputs() > 0 {
        source.push('\n');
    }

    // Layers
    for layer_idx in 0..network.nb_layers() {
        let Ok(layer) = network.layer(layer_idx) else {
            continue;
        };

        let layer_ptr = layer.inner.as_layer() as *const trtx_sys::nvinfer1::ILayer as usize;
        let line_start = source.len();

        // Output tensors: `out0(shape, dtype), out1(shape, dtype)`
        let n_out = layer.num_outputs(network);
        let mut has_net_output = false;
        for out_idx in 0..n_out {
            if out_idx > 0 {
                source.push_str(", ");
            }
            if let Ok(tensor) = layer.output(network, out_idx) {
                let ts = source.len();
                source.push_str(&fmt_tensor(network, tensor));
                let te = source.len();
                tensor_output_spans.insert(tensor.inner as usize, make_span(ts, te));
                tensor_producers.insert(tensor.inner as usize, layer_ptr);
                if net_output_ptrs.contains(&(tensor.inner as usize)) {
                    has_net_output = true;
                }
            }
        }

        // ` <- LayerType[layer_name](`
        let lt = fmt_layer_type(layer.layer_type_dynamic());
        let ln = layer.name(network);
        source.push_str(" <- ");
        source.push_str(&lt);
        source.push('[');
        source.push_str(&ln);
        source.push_str("](");
        layer_display.insert(layer_ptr, format!("{}[{}]", lt, ln));

        // Input tensors: `in0(shape, dtype), in1(shape, dtype)`
        let n_in = layer.num_inputs(network);
        let mut this_input_ptrs = Vec::new();
        let mut this_input_names = Vec::new();
        for in_idx in 0..n_in {
            if in_idx > 0 {
                source.push_str(", ");
            }
            if let Ok(tensor) = layer.input(network, in_idx) {
                source.push_str(&fmt_tensor(network, tensor));
                this_input_ptrs.push(tensor.inner as usize);
                this_input_names.push(tensor.name(network).unwrap_or_else(|_| "?".to_string()));
            }
        }

        source.push(')');
        if has_net_output {
            source.push_str("  // network output");
        }
        let line_end = source.len();
        source.push('\n');

        layer_spans.insert(layer_ptr, make_span(line_start, line_end));
        layer_input_tensor_ptrs.insert(layer_ptr, this_input_ptrs);
        layer_input_names.insert(layer_ptr, this_input_names);
    }

    NetworkText {
        source,
        layer_spans,
        layer_input_tensor_ptrs,
        layer_input_names,
        tensor_output_spans,
        tensor_producers,
        layer_display,
    }
}

// ---- diagnostic type -------------------------------------------------------

/// A [`miette::Diagnostic`] that wraps a [`crate::Error`] and points into an
/// annotated text dump of the [`NetworkDefinition`].
///
/// Obtain one via [`WithNetworkContext::with_problematic_layer`] or
/// [`WithNetworkContext::with_problematic_tensor`].
#[derive(Debug)]
pub struct NetworkDiagnostic {
    inner: Error,
    source_code: NamedSource<String>,
    primary_span: SourceSpan,
    primary_label: String,
    input_labels: Vec<LabeledSpan>,
    help_text: Option<String>,
}

impl NetworkDiagnostic {
    /// The annotated network text that miette will display as source context.
    pub fn source_text(&self) -> &str {
        self.source_code.inner()
    }

    /// Number of input labels attached (one per input tensor of the highlighted layer).
    pub fn num_input_labels(&self) -> usize {
        self.input_labels.len()
    }

    /// Byte length of the primary highlighted span.
    pub fn primary_span_len(&self) -> usize {
        self.primary_span.len()
    }

    /// Help text, if any.
    pub fn help(&self) -> Option<&str> {
        self.help_text.as_deref()
    }
}

impl fmt::Display for NetworkDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::error::Error for NetworkDiagnostic {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.inner)
    }
}

impl Diagnostic for NetworkDiagnostic {
    fn source_code(&self) -> Option<&dyn miette::SourceCode> {
        Some(&self.source_code)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        let primary = LabeledSpan::new_primary_with_span(
            Some(self.primary_label.to_string()),
            self.primary_span,
        );
        let inputs = self.input_labels.clone();
        Some(Box::new(std::iter::once(primary).chain(inputs)))
    }

    fn help<'a>(&'a self) -> Option<Box<dyn fmt::Display + 'a>> {
        self.help_text
            .as_deref()
            .map(|h| Box::new(h) as Box<dyn fmt::Display>)
    }
}

// ---- public trait ----------------------------------------------------------

/// Enrich a [`crate::Error`] with an annotated network dump for miette.
///
/// # Example
/// ```ignore
/// use trtx::miette_repr::WithNetworkContext;
///
/// let err = network.add_convolution(...)
///     .err()
///     .unwrap()
///     .with_problematic_layer(&network, &conv_layer);
/// println!("{:?}", miette::Report::new(err));
/// ```
pub trait WithNetworkContext {
    /// Annotate the error with the full network dump, highlighting `layer` and
    /// labelling each of its inputs.
    fn with_problematic_layer<'n, Inner: AsLayer>(
        self,
        network: &'n NetworkDefinition,
        layer: &Layer<'n, Inner>,
        layer_message: &str,
    ) -> NetworkDiagnostic;

    /// Annotate the error with the full network dump, highlighting `tensor` at
    /// its definition site and labelling the inputs of the layer that produced it.
    fn with_problematic_tensor<'n>(
        self,
        network: &'n NetworkDefinition,
        tensor: Tensor<'n>,
        tensor_message: &str,
    ) -> NetworkDiagnostic;
}

/// For each input tensor of `layer_ptr`, return a label pointing to where that
/// tensor is *defined* (network-input line or the output span of its producing layer).
fn input_labels_at_definition(nt: &NetworkText, layer_ptr: usize) -> Vec<LabeledSpan> {
    let ptrs = match nt.layer_input_tensor_ptrs.get(&layer_ptr) {
        Some(v) => v,
        None => return vec![],
    };
    let names = nt.layer_input_names.get(&layer_ptr);
    ptrs.iter()
        .enumerate()
        .filter_map(|(i, &tptr)| {
            let span = nt.tensor_output_spans.get(&tptr).copied()?;
            let name = names
                .and_then(|n| n.get(i))
                .map(String::as_str)
                .unwrap_or("?");
            Some(LabeledSpan::new_with_span(
                Some(format!("input {i}: {name}")),
                span,
            ))
        })
        .collect()
}

impl WithNetworkContext for Error {
    fn with_problematic_layer<'n, Inner: AsLayer>(
        self,
        network: &'n NetworkDefinition,
        layer: &Layer<'n, Inner>,
        layer_message: &str,
    ) -> NetworkDiagnostic {
        let nt = build_network_text(network);
        let layer_ptr = layer.inner.as_layer() as *const trtx_sys::nvinfer1::ILayer as usize;

        let (primary_span, input_labels, help_text) = match nt.layer_spans.get(&layer_ptr).copied()
        {
            Some(lspan) => {
                let labels = input_labels_at_definition(&nt, layer_ptr);
                let help = nt
                    .layer_input_names
                    .get(&layer_ptr)
                    .map(|names| format!("Layer inputs: {}", names.join(", ")));
                (lspan, labels, help)
            }
            None => (make_span(0, 0), vec![], None),
        };

        NetworkDiagnostic {
            inner: self,
            source_code: NamedSource::new("network", nt.source),
            primary_span,
            primary_label: layer_message.to_string(),
            input_labels,
            help_text,
        }
    }

    fn with_problematic_tensor<'n>(
        self,
        network: &'n NetworkDefinition,
        tensor: Tensor<'n>,
        tensor_message: &str,
    ) -> NetworkDiagnostic {
        let nt = build_network_text(network);
        let tensor_ptr = tensor.inner as usize;

        let primary_span = nt
            .tensor_output_spans
            .get(&tensor_ptr)
            .copied()
            .unwrap_or_else(|| make_span(0, 0));

        let (input_labels, help_text) = match nt.tensor_producers.get(&tensor_ptr).copied() {
            Some(layer_ptr) => {
                let labels = input_labels_at_definition(&nt, layer_ptr);
                let producer = nt
                    .layer_display
                    .get(&layer_ptr)
                    .map(String::as_str)
                    .unwrap_or("?");
                let inputs = nt
                    .layer_input_names
                    .get(&layer_ptr)
                    .map(|n| n.join(", "))
                    .unwrap_or_default();
                let help = format!("Produced by {}. Inputs: {}", producer, inputs);
                (labels, Some(help))
            }
            None => (vec![], Some("This tensor is a network input".to_string())),
        };

        NetworkDiagnostic {
            inner: self,
            source_code: NamedSource::new("network", nt.source),
            primary_span,
            primary_label: tensor_message.to_string(),
            input_labels,
            help_text,
        }
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use miette::GraphicalReportHandler;
    use trtx_sys::{DataType, LayerType};

    // ---- pure formatting helpers (no TensorRT required) --------------------

    #[test]
    fn fmt_dtype_common_types() {
        assert_eq!(fmt_dtype(DataType::kFLOAT), "F32");
        assert_eq!(fmt_dtype(DataType::kHALF), "F16");
        assert_eq!(fmt_dtype(DataType::kBF16), "BF16");
        assert_eq!(fmt_dtype(DataType::kINT8), "I8");
        assert_eq!(fmt_dtype(DataType::kINT32), "I32");
        assert_eq!(fmt_dtype(DataType::kINT64), "I64");
        assert_eq!(fmt_dtype(DataType::kBOOL), "Bool");
    }

    #[test]
    fn fmt_dims_static() {
        assert_eq!(fmt_dims(&[1, 3, 224, 224]), "[1,3,224,224]");
    }

    #[test]
    fn fmt_dims_dynamic() {
        assert_eq!(fmt_dims(&[-1, 3, 224, 224]), "[?,3,224,224]");
    }

    #[test]
    fn fmt_dims_scalar() {
        assert_eq!(fmt_dims(&[]), "[]");
    }

    #[test]
    fn fmt_layer_type_strips_k_and_lowercases() {
        assert_eq!(fmt_layer_type(LayerType::kCONVOLUTION), "Convolution");
        assert_eq!(fmt_layer_type(LayerType::kACTIVATION), "Activation");
        assert_eq!(fmt_layer_type(LayerType::kELEMENTWISE), "Elementwise");
    }

    // ---- integration tests (require TensorRT + CUDA) -----------------------

    #[test]
    #[cfg(not(feature = "mock_runtime"))]
    fn with_problematic_layer_source_contains_layer() {
        use crate::{Builder, ElementWiseOperation, Logger};

        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let mut network = builder.create_network(0).unwrap();

        let input = network.add_input("data", DataType::kFLOAT, &[4]).unwrap();
        let one_bytes = 1.0f32.to_le_bytes();
        let one_layer = network
            .add_small_constant_copied(&[1], &one_bytes, DataType::kFLOAT)
            .unwrap();
        let one_t = one_layer.output(&network, 0).unwrap();
        let mut sum = network
            .add_elementwise(&input, &one_t, ElementWiseOperation::kSUM)
            .unwrap();
        sum.set_name(&mut network, "add_one").unwrap();

        let diag = Error::Runtime("test".into()).with_problematic_layer(
            &network,
            &sum,
            "This layer caused the problem",
        );

        let mut out = String::new();
        let handler = GraphicalReportHandler::new_themed(miette::GraphicalTheme::unicode())
            .with_context_lines(6);
        handler.render_report(&mut out, &diag).unwrap();
        assert_snapshot!(out);
        println!("{out}");
    }

    #[test]
    #[cfg(not(feature = "mock_runtime"))]
    fn with_problematic_tensor_points_to_producer() {
        use crate::{Builder, ElementWiseOperation, Logger};

        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let mut network = builder.create_network(0).unwrap();

        let input = network.add_input("data", DataType::kFLOAT, &[4]).unwrap();
        let one_bytes = 1.0f32.to_le_bytes();
        let one_layer = network
            .add_small_constant_copied(&[1], &one_bytes, DataType::kFLOAT)
            .unwrap();
        let one_t = one_layer.output(&network, 0).unwrap();
        let mut sum = network
            .add_elementwise(&input, &one_t, ElementWiseOperation::kSUM)
            .unwrap();
        sum.set_name(&mut network, "add_one").unwrap();
        let out = sum.output(&network, 0).unwrap();

        let diag = Error::Runtime("test".into()).with_problematic_tensor(
            &network,
            out,
            "problematic tensor",
        );

        let mut out = String::new();
        let handler = GraphicalReportHandler::new_themed(miette::GraphicalTheme::unicode())
            .with_context_lines(6);
        handler.render_report(&mut out, &diag).unwrap();
        assert_snapshot!(out);
        println!("{out}");
    }

    #[test]
    #[cfg(not(feature = "mock_runtime"))]
    fn with_problematic_tensor_network_input() {
        use crate::{Builder, Logger};

        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let mut network = builder.create_network(0).unwrap();

        let input = network
            .add_input("img", DataType::kFLOAT, &[1, 3, 224, 224])
            .unwrap();

        let diag = Error::Runtime("test".into()).with_problematic_tensor(
            &network,
            input,
            "This tensor caused the problem",
        );

        let mut out = String::new();
        let handler = GraphicalReportHandler::new_themed(miette::GraphicalTheme::unicode());
        handler.render_report(&mut out, &diag).unwrap();
        assert_snapshot!(out);
        println!("{out}");
    }
}
