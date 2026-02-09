//! Helper functions for converting TensorRT enums to strings
//!
//! This module provides helper functions for getting string names of TensorRT enum variants.
//! These are useful for error messages, debugging, and logging.

use trtx_sys::nvinfer1::{
    ActivationType, CumulativeOperation, DataType, ElementWiseOperation, GatherMode,
    InterpolationMode, PoolingType, ReduceOperation, ResizeCoordinateTransformation,
    ResizeRoundMode, ResizeSelector, ScatterMode, UnaryOperation,
};

// ============================================================================
// DataType
// ============================================================================

/// Get the string name of a DataType enum variant
pub fn datatype_name(dt: &DataType) -> &'static str {
    match *dt {
        DataType::kFLOAT => "kFLOAT",
        DataType::kHALF => "kHALF",
        DataType::kINT8 => "kINT8",
        DataType::kINT32 => "kINT32",
        DataType::kUINT8 => "kUINT8",
        DataType::kBOOL => "kBOOL",
        DataType::kFP8 => "kFP8",
        DataType::kBF16 => "kBF16",
        DataType::kINT64 => "kINT64",
        DataType::kINT4 => "kINT4",
        DataType::kFP4 => "kFP4",
        DataType::kE8M0 => "kE8M0",
    }
}

// ============================================================================
// ElementWiseOperation
// ============================================================================

/// Get the string name of an ElementWiseOperation enum variant
pub fn elementwise_op_name(op: &ElementWiseOperation) -> &'static str {
    match *op {
        ElementWiseOperation::kSUM => "kSUM",
        ElementWiseOperation::kPROD => "kPROD",
        ElementWiseOperation::kMAX => "kMAX",
        ElementWiseOperation::kMIN => "kMIN",
        ElementWiseOperation::kSUB => "kSUB",
        ElementWiseOperation::kDIV => "kDIV",
        ElementWiseOperation::kPOW => "kPOW",
        ElementWiseOperation::kFLOOR_DIV => "kFLOOR_DIV",
        ElementWiseOperation::kAND => "kAND",
        ElementWiseOperation::kOR => "kOR",
        ElementWiseOperation::kXOR => "kXOR",
        ElementWiseOperation::kEQUAL => "kEQUAL",
        ElementWiseOperation::kGREATER => "kGREATER",
        ElementWiseOperation::kLESS => "kLESS",
    }
}

// ============================================================================
// UnaryOperation
// ============================================================================

/// Get the string name of a UnaryOperation enum variant
pub fn unary_op_name(op: &UnaryOperation) -> &'static str {
    match *op {
        UnaryOperation::kEXP => "kEXP",
        UnaryOperation::kLOG => "kLOG",
        UnaryOperation::kSQRT => "kSQRT",
        UnaryOperation::kRECIP => "kRECIP",
        UnaryOperation::kABS => "kABS",
        UnaryOperation::kNEG => "kNEG",
        UnaryOperation::kSIN => "kSIN",
        UnaryOperation::kCOS => "kCOS",
        UnaryOperation::kTAN => "kTAN",
        UnaryOperation::kSINH => "kSINH",
        UnaryOperation::kCOSH => "kCOSH",
        UnaryOperation::kASIN => "kASIN",
        UnaryOperation::kACOS => "kACOS",
        UnaryOperation::kATAN => "kATAN",
        UnaryOperation::kASINH => "kASINH",
        UnaryOperation::kACOSH => "kACOSH",
        UnaryOperation::kATANH => "kATANH",
        UnaryOperation::kCEIL => "kCEIL",
        UnaryOperation::kFLOOR => "kFLOOR",
        UnaryOperation::kERF => "kERF",
        UnaryOperation::kNOT => "kNOT",
        UnaryOperation::kSIGN => "kSIGN",
        UnaryOperation::kROUND => "kROUND",
        UnaryOperation::kISINF => "kISINF",
        UnaryOperation::kISNAN => "kISNAN",
    }
}

// ============================================================================
// ActivationType
// ============================================================================

/// Get the string name of an ActivationType enum variant
pub fn activation_type_name(at: &ActivationType) -> &'static str {
    match *at {
        ActivationType::kRELU => "kRELU",
        ActivationType::kSIGMOID => "kSIGMOID",
        ActivationType::kTANH => "kTANH",
        ActivationType::kLEAKY_RELU => "kLEAKY_RELU",
        ActivationType::kELU => "kELU",
        ActivationType::kSELU => "kSELU",
        ActivationType::kSOFTSIGN => "kSOFTSIGN",
        ActivationType::kSOFTPLUS => "kSOFTPLUS",
        ActivationType::kCLIP => "kCLIP",
        ActivationType::kHARD_SIGMOID => "kHARD_SIGMOID",
        ActivationType::kSCALED_TANH => "kSCALED_TANH",
        ActivationType::kTHRESHOLDED_RELU => "kTHRESHOLDED_RELU",
        ActivationType::kGELU_ERF => "kGELU_ERF",
        ActivationType::kGELU_TANH => "kGELU_TANH",
    }
}

// ============================================================================
// PoolingType
// ============================================================================

/// Get the string name of a PoolingType enum variant
pub fn pooling_type_name(pt: &PoolingType) -> &'static str {
    match *pt {
        PoolingType::kMAX => "kMAX",
        PoolingType::kAVERAGE => "kAVERAGE",
        PoolingType::kMAX_AVERAGE_BLEND => "kMAX_AVERAGE_BLEND",
    }
}

// ============================================================================
// ReduceOperation
// ============================================================================

/// Get the string name of a ReduceOperation enum variant
pub fn reduce_op_name(op: &ReduceOperation) -> &'static str {
    match *op {
        ReduceOperation::kSUM => "kSUM",
        ReduceOperation::kPROD => "kPROD",
        ReduceOperation::kMAX => "kMAX",
        ReduceOperation::kMIN => "kMIN",
        ReduceOperation::kAVG => "kAVG",
    }
}

// ============================================================================
// CumulativeOperation
// ============================================================================

/// Get the string name of a CumulativeOperation enum variant
pub fn cumulative_op_name(op: &CumulativeOperation) -> &'static str {
    match *op {
        CumulativeOperation::kSUM => "kSUM",
        #[cfg(feature = "mock")]
        CumulativeOperation::kPROD => "kPROD",
        #[cfg(feature = "mock")]
        CumulativeOperation::kMIN => "kMIN",
        #[cfg(feature = "mock")]
        CumulativeOperation::kMAX => "kMAX",
    }
}

// ============================================================================
// GatherMode
// ============================================================================

/// Get the string name of a GatherMode enum variant
pub fn gather_mode_name(mode: &GatherMode) -> &'static str {
    match *mode {
        GatherMode::kDEFAULT => "kDEFAULT",
        GatherMode::kELEMENT => "kELEMENT",
        GatherMode::kND => "kND",
    }
}

// ============================================================================
// ScatterMode
// ============================================================================

/// Get the string name of a ScatterMode enum variant
pub fn scatter_mode_name(mode: &ScatterMode) -> &'static str {
    match *mode {
        ScatterMode::kELEMENT => "kELEMENT",
        ScatterMode::kND => "kND",
    }
}

// ============================================================================
// InterpolationMode (ResizeMode)
// ============================================================================

/// Get the string name of an InterpolationMode enum variant
/// Note: ResizeMode is a typedef for InterpolationMode
pub fn interpolation_mode_name(mode: &InterpolationMode) -> &'static str {
    match *mode {
        InterpolationMode::kNEAREST => "kNEAREST",
        InterpolationMode::kLINEAR => "kLINEAR",
        InterpolationMode::kCUBIC => "kCUBIC",
    }
}

// ============================================================================
// ResizeCoordinateTransformation
// ============================================================================

/// Get the string name of a ResizeCoordinateTransformation enum variant
pub fn resize_coord_transform_name(transform: &ResizeCoordinateTransformation) -> &'static str {
    match *transform {
        ResizeCoordinateTransformation::kALIGN_CORNERS => "kALIGN_CORNERS",
        ResizeCoordinateTransformation::kASYMMETRIC => "kASYMMETRIC",
        ResizeCoordinateTransformation::kHALF_PIXEL => "kHALF_PIXEL",
        #[cfg(feature = "mock")]
        ResizeCoordinateTransformation::kHALF_PIXEL_SYMMETRIC => "kHALF_PIXEL_SYMMETRIC",
    }
}

// ============================================================================
// ResizeSelector
// ============================================================================

/// Get the string name of a ResizeSelector enum variant
pub fn resize_selector_name(selector: &ResizeSelector) -> &'static str {
    match *selector {
        ResizeSelector::kFORMULA => "kFORMULA",
        #[cfg(feature = "mock")]
        ResizeSelector::kSIZES => "kSIZES",
        ResizeSelector::kUPPER => "kUPPER",
    }
}

// ============================================================================
// ResizeRoundMode
// ============================================================================

/// Get the string name of a ResizeRoundMode enum variant
pub fn resize_round_mode_name(mode: &ResizeRoundMode) -> &'static str {
    match *mode {
        ResizeRoundMode::kFLOOR => "kFLOOR",
        ResizeRoundMode::kCEIL => "kCEIL",
        #[cfg(feature = "mock")]
        ResizeRoundMode::kROUND => "kROUND",
        ResizeRoundMode::kHALF_UP => "kHALF_UP",
        ResizeRoundMode::kHALF_DOWN => "kHALF_DOWN",
    }
}

// NOTE: RNN enum helper functions removed - IRNNv2Layer is deprecated and bindings unavailable

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datatype_name() {
        assert_eq!(datatype_name(&DataType::kFLOAT), "kFLOAT");
        assert_eq!(datatype_name(&DataType::kBOOL), "kBOOL");
    }

    #[test]
    fn test_elementwise_op_name() {
        assert_eq!(elementwise_op_name(&ElementWiseOperation::kSUM), "kSUM");
        assert_eq!(
            elementwise_op_name(&ElementWiseOperation::kGREATER),
            "kGREATER"
        );
    }

    #[test]
    fn test_unary_op_name() {
        assert_eq!(unary_op_name(&UnaryOperation::kEXP), "kEXP");
        assert_eq!(unary_op_name(&UnaryOperation::kNOT), "kNOT");
    }

    #[test]
    fn test_activation_type_name() {
        assert_eq!(activation_type_name(&ActivationType::kRELU), "kRELU");
        assert_eq!(
            activation_type_name(&ActivationType::kGELU_ERF),
            "kGELU_ERF"
        );
    }

    #[test]
    fn test_pooling_type_name() {
        assert_eq!(pooling_type_name(&PoolingType::kMAX), "kMAX");
        assert_eq!(pooling_type_name(&PoolingType::kAVERAGE), "kAVERAGE");
    }

    #[test]
    fn test_reduce_op_name() {
        assert_eq!(reduce_op_name(&ReduceOperation::kSUM), "kSUM");
        assert_eq!(reduce_op_name(&ReduceOperation::kAVG), "kAVG");
    }
}
