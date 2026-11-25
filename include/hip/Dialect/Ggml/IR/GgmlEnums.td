//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT_GGML_IR_GGMLENUMS_TD
#define HIP_DIALECT_GGML_IR_GGMLENUMS_TD

include "mlir/IR/EnumAttr.td"

/// One value of a GGML enumeration, and what the dialect calls it.
///
/// GGML's ordinal is not the dialect's -- `GGML_PREC_F32` is ten and the
/// dialect's `f32` is one -- so the correspondence is the enumerator named
/// here rather than a number that happens to line up.
class Ggml_EnumCase<string symbol, int value, string spelling, string ggml>
    : I32EnumAttrCase<symbol, value, spelling> {
  string ggmlEnumerator = ggml;
}

/// A dialect enumeration the capture reads out of GGML. `ggmlType` is the C
/// type a value arrives as and `ggmlSpelling` the function that names one;
/// `ggmlExhausted` is the enumerator that bounds the C enumeration, so that
/// the switch stays exhaustive and a value llama.cpp adds is a warning.
class Ggml_GgmlEnum<string name, string summary, list<Ggml_EnumCase> cases,
                    string type, string spelling, string exhausted = "">
    : I32EnumAttr<name, summary, cases> {
  string ggmlType = type;
  string ggmlSpelling = spelling;
  string ggmlExhausted = exhausted;
}

def Ggml_ComputePrecisionDefault :
    Ggml_EnumCase<"Default", 0, "default", "GGML_PREC_UNDEFINED">;
def Ggml_ComputePrecisionF32 : Ggml_EnumCase<"F32", 1, "f32", "GGML_PREC_F32">;

def Ggml_ComputePrecision : Ggml_GgmlEnum<
    "ComputePrecision", "requested computation precision",
    [Ggml_ComputePrecisionDefault, Ggml_ComputePrecisionF32], "ggml_prec", "compute_precision_name"> {
  let cppNamespace = "::mlir::ggml";
  let genSpecializedAttr = 0;
}

def Ggml_RopeModeNormal : Ggml_EnumCase<"Normal", 0, "normal", "GGML_ROPE_TYPE_NORMAL">;
def Ggml_RopeModeNeox : Ggml_EnumCase<"Neox", 1, "neox", "GGML_ROPE_TYPE_NEOX">;
def Ggml_RopeModeMrope : Ggml_EnumCase<"Mrope", 2, "mrope", "GGML_ROPE_TYPE_MROPE">;
def Ggml_RopeModeVision : Ggml_EnumCase<"Vision", 3, "vision", "GGML_ROPE_TYPE_VISION">;
def Ggml_RopeModeImrope : Ggml_EnumCase<"Imrope", 4, "imrope", "GGML_ROPE_TYPE_IMROPE">;

def Ggml_RopeMode : Ggml_GgmlEnum<
    "RopeMode", "rotary embedding layout",
    [Ggml_RopeModeNormal, Ggml_RopeModeNeox, Ggml_RopeModeMrope,
     Ggml_RopeModeVision, Ggml_RopeModeImrope], "int", "rope_mode_name"> {
  let cppNamespace = "::mlir::ggml";
  let genSpecializedAttr = 0;
}

def Ggml_GluKindReglu : Ggml_EnumCase<"Reglu", 0, "reglu", "GGML_GLU_OP_REGLU">;
def Ggml_GluKindGeglu : Ggml_EnumCase<"Geglu", 1, "geglu", "GGML_GLU_OP_GEGLU">;
def Ggml_GluKindSwiglu : Ggml_EnumCase<"Swiglu", 2, "swiglu", "GGML_GLU_OP_SWIGLU">;
def Ggml_GluKindSwigluOai :
    Ggml_EnumCase<"SwigluOai", 3, "swiglu_oai", "GGML_GLU_OP_SWIGLU_OAI">;
def Ggml_GluKindGegluErf :
    Ggml_EnumCase<"GegluErf", 4, "geglu_erf", "GGML_GLU_OP_GEGLU_ERF">;
def Ggml_GluKindGegluQuick :
    Ggml_EnumCase<"GegluQuick", 5, "geglu_quick", "GGML_GLU_OP_GEGLU_QUICK">;

def Ggml_GluKind : Ggml_GgmlEnum<
    "GluKind", "gated linear-unit activation",
    [Ggml_GluKindReglu, Ggml_GluKindGeglu, Ggml_GluKindSwiglu,
     Ggml_GluKindSwigluOai, Ggml_GluKindGegluErf,
     Ggml_GluKindGegluQuick], "ggml_glu_op", "glu_kind_name", "GGML_GLU_OP_COUNT"> {
  let cppNamespace = "::mlir::ggml";
  let genSpecializedAttr = 0;
}

#endif // HIP_DIALECT_GGML_IR_GGMLENUMS_TD
