//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===- GgmlSource.td - Where GGML keeps an operation's configuration ---===//
//
// An operation's arguments say what it was configured with. GGML keeps those
// values in `op_params`, sixty-four bytes read as int32 or float32 at a fixed
// index, and a few of them somewhere else entirely. That is the one thing the
// argument list cannot say, so it is said here.
//
// From it `ggml-tblgen` generates everything that walks those values: the
// struct the capture records them in, the decoding out of `op_params`, and
// the attributes the dialect operation carries. A value is therefore declared
// once, and adding one to an operation is one line.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_IR_GGMLSOURCE_TD
#define HIP_DIALECT_GGML_IR_GGMLSOURCE_TD

/// Where one configured value comes from.
///
/// `argument` names the operation argument it becomes, which is where its name
/// and its attribute type come from. `member` is what the capture's struct
/// calls it; by default that is the argument in camel case, and it is given
/// only where GGML's name for something reads badly as a member. `attribute`
/// is zero for a value the result type already states, which the capture
/// records and the emitter checks rather than repeating on the operation.
class GgmlParam<string argument_, string member_ = "", bit attribute_ = 1> {
  string argument = argument_;
  string member = member_;
  bit attribute = attribute_;
}

/// An int32 at `index`, which becomes an `I64Attr`.
class FromI32<string a, int index_, string m = ""> : GgmlParam<a, m> {
  int index = index_;
}

/// A float32 at `index`, which becomes an `F32Attr`.
class FromF32<string a, int index_, string m = ""> : GgmlParam<a, m> {
  int index = index_;
}

/// A nonzero int32 at `index`, which becomes a `BoolAttr`.
class FromI32Flag<string a, int index_, string m = ""> : GgmlParam<a, m> {
  int index = index_;
}

/// One bit in the int32 at `index`, which becomes a `BoolAttr`.
class FromI32Bit<string a, int index_, int mask_, string m = "">
    : GgmlParam<a, m> {
  int index = index_;
  int mask = mask_;
}

/// `count` consecutive int32 values beginning at `index`, which become one
/// array attribute.
class FromI32Run<string a, int index_, int count_, string m = "">
    : GgmlParam<a, m> {
  int index = index_;
  int count = count_;
}

/// An enumerator GGML spells as an int32 ordinal at `index`. `spelling` names
/// it, and `cast` is the GGML enum the ordinal belongs to.
class FromI32Enum<string a, int index_, string spelling_, string cast_,
                  string m = ""> : GgmlParam<a, m> {
  int index = index_;
  string spelling = spelling_;
  string cast = cast_;
}

/// An enumerator stored in selected bits of the int32 at `index`.
class FromI32MaskedEnum<string a, int index_, int mask_, string spelling_,
                        string cast_, string m = ""> : GgmlParam<a, m> {
  int index = index_;
  int mask = mask_;
  string spelling = spelling_;
  string cast = cast_;
}

/// A value `op_params` does not hold. `expression` reads it off the tensor,
/// which is in scope as `tensor`.
class FromTensor<string a, string type_, string expression_, string m = "",
                 bit attribute_ = 1> : GgmlParam<a, m, attribute_> {
  string type = type_;
  string expression = expression_;
}

/// An enumerator GGML hands over through a call rather than through
/// `op_params`. `expression` is that call and `spelling` names the result.
class FromTensorEnum<string a, string expression_, string spelling_,
                     string m = ""> : GgmlParam<a, m> {
  string expression = expression_;
  string spelling = spelling_;
}

/// A value GGML does not store, because this spelling of the operation fixes
/// it. `expression` is the value, written as C++.
///
/// This only decodes: the member it fills is declared by the spelling that
/// reads the value out of `op_params`, and the two share one struct.
class Fixed<string a, string expression_, string m = ""> : GgmlParam<a, m> {
  string expression = expression_;
}

/// A value that is not GGML's at all: whether the operation was given the
/// optional operand at `slot`.
///
/// GGML says this by leaving a slot empty, and the dialect says it with an
/// argument, because an operation whose operand list is a fixed-width array
/// cannot say it any other way. Nothing about it reaches the capture, so it is
/// declared apart from the values that do.
class FromOperandPresence<string argument_, int slot_> {
  string argument = argument_;
  int slot = slot_;
}

/// A second GGML operation this one is, and the configuration that spelling
/// implies. GGML has two names for one operation when one is a fixed instance
/// of the other, and the dialect keeps the general one.
class GgmlSpelling<string operation_, list<GgmlParam> sources_ = []> {
  string operation = operation_;
  list<GgmlParam> sources = sources_;
}

#endif // HIP_DIALECT_GGML_IR_GGMLSOURCE_TD
