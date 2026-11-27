//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT_GGML_IR_GGMLENUMS_TD
#define HIP_DIALECT_GGML_IR_GGMLENUMS_TD

include "mlir/IR/EnumAttr.td"

/// One value of a GGML enumeration, and what the dialect calls it.
///
/// GGML's number is not the dialect's -- `GGML_PREC_F32` is ten and the
/// dialect's `f32` is one -- so each case says both: `ggmlValue` is the number
/// GGML stores in `op_params`, `ggml` the enumerator that number is.
class Ggml_EnumCase<string symbol, int value, string spelling, string ggml,
                    int ggmlValue_>
    : I32EnumAttrCase<symbol, value, spelling> {
  string ggmlEnumerator = ggml;
  int ggmlValue = ggmlValue_;
}

/// A dialect enumeration the capture reads out of GGML. `ggmlSpelling` is
/// the function that names a value; one this does not list is refused.
class Ggml_GgmlEnum<string name, string summary, list<Ggml_EnumCase> cases,
                    string spelling>
    : I32EnumAttr<name, summary, cases> {
  string ggmlSpelling = spelling;
}

def Ggml_ComputePrecisionDefault :
    Ggml_EnumCase<"Default", 0, "default", "GGML_PREC_UNDEFINED", 0>;
def Ggml_ComputePrecisionF32 : Ggml_EnumCase<"F32", 1, "f32", "GGML_PREC_F32", 10>;

def Ggml_ComputePrecision : Ggml_GgmlEnum<
    "ComputePrecision", "requested computation precision",
    [Ggml_ComputePrecisionDefault, Ggml_ComputePrecisionF32], "compute_precision_name"> {
  let cppNamespace = "::mlir::ggml";
  let genSpecializedAttr = 0;
}

def Ggml_RopeModeNormal : Ggml_EnumCase<"Normal", 0, "normal", "GGML_ROPE_TYPE_NORMAL", 0>;
def Ggml_RopeModeNeox : Ggml_EnumCase<"Neox", 1, "neox", "GGML_ROPE_TYPE_NEOX", 2>;
def Ggml_RopeModeMrope : Ggml_EnumCase<"Mrope", 2, "mrope", "GGML_ROPE_TYPE_MROPE", 8>;
def Ggml_RopeModeVision : Ggml_EnumCase<"Vision", 3, "vision", "GGML_ROPE_TYPE_VISION", 24>;
def Ggml_RopeModeImrope : Ggml_EnumCase<"Imrope", 4, "imrope", "GGML_ROPE_TYPE_IMROPE", 40>;

def Ggml_RopeMode : Ggml_GgmlEnum<
    "RopeMode", "rotary embedding layout",
    [Ggml_RopeModeNormal, Ggml_RopeModeNeox, Ggml_RopeModeMrope,
     Ggml_RopeModeVision, Ggml_RopeModeImrope], "rope_mode_name"> {
  let cppNamespace = "::mlir::ggml";
  let genSpecializedAttr = 0;
}

def Ggml_GluKindReglu : Ggml_EnumCase<"Reglu", 0, "reglu", "GGML_GLU_OP_REGLU", 0>;
def Ggml_GluKindGeglu : Ggml_EnumCase<"Geglu", 1, "geglu", "GGML_GLU_OP_GEGLU", 1>;
def Ggml_GluKindSwiglu : Ggml_EnumCase<"Swiglu", 2, "swiglu", "GGML_GLU_OP_SWIGLU", 2>;
def Ggml_GluKindSwigluOai :
    Ggml_EnumCase<"SwigluOai", 3, "swiglu_oai", "GGML_GLU_OP_SWIGLU_OAI", 3>;
def Ggml_GluKindGegluErf :
    Ggml_EnumCase<"GegluErf", 4, "geglu_erf", "GGML_GLU_OP_GEGLU_ERF", 4>;
def Ggml_GluKindGegluQuick :
    Ggml_EnumCase<"GegluQuick", 5, "geglu_quick", "GGML_GLU_OP_GEGLU_QUICK", 5>;

def Ggml_GluKind : Ggml_GgmlEnum<
    "GluKind", "gated linear-unit activation",
    [Ggml_GluKindReglu, Ggml_GluKindGeglu, Ggml_GluKindSwiglu,
     Ggml_GluKindSwigluOai, Ggml_GluKindGegluErf,
     Ggml_GluKindGegluQuick], "glu_kind_name"> {
  let cppNamespace = "::mlir::ggml";
  let genSpecializedAttr = 0;
}

#endif // HIP_DIALECT_GGML_IR_GGMLENUMS_TD
