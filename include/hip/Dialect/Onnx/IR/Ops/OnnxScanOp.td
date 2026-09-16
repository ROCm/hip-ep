//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_ScanOp
//===----------------------------------------------------------------------===//

def Onnx_ScanOp : Onnx_Op<"Scan", [RecursiveMemoryEffects]> {
  let summary = "ONNX Scan, schema version 25";

  let description = [{
    Scan can be used to iterate over one or more scan_input tensors,
    constructing zero or more scan_output tensors. It combines ideas from general recurrences,
    functional programming constructs such as scan, fold, map, and zip, and is intended to enable
    generalizations of RNN-like constructs for sequence-to-sequence processing.
    Other tensors (referred to as state_variables here) can be used to carry a state
    when iterating from one element to another (similar to hidden-state in RNNs, also referred
    to as loop-carried dependences in the context of loops).
    Many common usages involve a single scan_input tensor (where functionality
    similar to scan, fold and map can be obtained). When more than one scan_input is used,
    a behavior similar to zip is obtained.

    The attribute body must be a graph, specifying the computation to be performed in
    every iteration. It takes as input the current values of the state_variables and
    the current iterated element of the scan_inputs. It must return the (updated) values
    of the state_variables and zero or more scan_output_element tensors. The values of the
    scan_output_element tensors are concatenated over all the iterations to produce the
    scan_output values of the scan construct (similar to the concatenated intermediate
    hidden-state values of RNN-like constructs). All the output tensors (state_variables as
    well as scan_output_element tensors) are required to have the same shape in each iteration
    of the loop (a restriction imposed to enable efficient memory allocation).

    Note that the iterated element passed to the body subgraph does not have a sequence
    axis. It will have a rank one less than the rank of the corresponding scan_input.

    The scan operation returns the final values of the state_variables as well as the
    scan_outputs.

    The optional attribute scan_input_directions specifies the direction (forward or backward)
    for each scan input. If this attribute is omitted, all sequences are scanned in the forward
    direction. A bidirectional scan may be performed by specifying the same tensor input twice
    in the scan_inputs, once with a forward direction, and once with a backward direction.

    The scan_output of the operation is produced by concatenating the scan_output_element
    values produced by the body in each iteration.  The optional attribute scan_output_directions
    specifies the direction in which scan_output is constructed (by appending or prepending the
    scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
    is omitted, the scan_output_element is appended to the scan_output in each iteration.

    The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
    If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
    batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
    Note that scanning a non-zero axis may be less efficient than scanning axis zero.

    The optional attribute scan_output_axes specifies the axis along which the scan_outputs
    are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
    scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
    value of 1.

    Note that because of the ONNX restriction that only the last parameter of an operator can
    be variadic, the initial-states and scan-inputs are listed together as one input parameter.
    Similarly, the final-states and scan-outputs are listed together as one output parameter.
    The attribute num_scan_inputs indicates the number M of scan-inputs.

    The behavior of

        Scan <
            num_scan_inputs = m,
            body = loop-body,
            scan_input_axes = [axis_1, ..., axis_m]
        > (init_1, ..., init_n, scan_1, ..., scan_m)

    is equivalent to the following pseudo-code:

        // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
        // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
        sequence_length = scan_1.shape[axis_1];

        // initialize state-variables
        st_1 = init_1; ... st_n = init_n;
        // initialize scan-output variables: [] denotes an empty tensor
        scan_out_1 = []; ...; scan_out_k = [];
        // identify number of iterations:

        // execute loop
        for (int t = 0; t < sequence_length; ++t) {
            // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
            // of rank one less than T obtained by indexing T at position t along axis k.
            si_1 = scan_1<axis=axis_1>[t];
            ... ;
            si_m = scan_m<axis=axis_m>[t];
            // execute loop-body
            st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
            // accumulate the scan-output elements
            scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
        }

        return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

    *Sample usage: Encoding RNN using a Scan*

    The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
    recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
    be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
    %Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
    values are computed in the outer graph, they need to be passed in as extra state_variables.

        graph rnn-encoding {
          %H_0 = ...
          %X = ...
          %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
          return %Y, %Y_h
        }

        graph rnn-cell-1 (
          %H_tminus1[FLOAT, tensor]
          %X_t[FLOAT, tensor]
        ) {
          %Wi = ...
          %Ri = ...
          %Wbi = ...
          %Rbi = ...
          %t1 = X_t * (Wi^T)
          %t2 = H_tminus1*(Ri^T)
          %t3 = Add(%t1, %t2)
          %t4 = Add(%t3, %Wbi)
          %t5 = Add(%t4, %Rbi)
          %Ht = Tanh(%t5)
          %Accumulate = Identity(%Ht)
          return %Ht, %Accumulate
        }
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>
    ]>>:$initial_state_and_scan_inputs,
    SI64Attr:$num_scan_inputs,
    OptionalAttr<I64ArrayAttr>:$scan_input_axes,
    OptionalAttr<I64ArrayAttr>:$scan_input_directions,
    OptionalAttr<I64ArrayAttr>:$scan_output_axes,
    OptionalAttr<I64ArrayAttr>:$scan_output_directions
  );

  let results = (outs
    Variadic<AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>
    ]>>:$final_state_and_scan_outputs
  );

  let regions = (region AnyRegion:$body);
}
