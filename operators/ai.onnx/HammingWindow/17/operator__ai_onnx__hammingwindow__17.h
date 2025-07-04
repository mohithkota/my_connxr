//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__HAMMINGWINDOW__17_H
# define OPERATOR_OPERATOR__AI_ONNX__HAMMINGWINDOW__17_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'HammingWindow' version 17
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106.
 * 
 * Constraint T1:
 *   Constrain the input size to int64_t.
 *   Allowed Types: tensor_int32, tensor_int64
 * 
 * Constraint T2:
 *   Constrain output types to numeric tensors.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_uint16, tensor_uint32, tensor_uint64,
 *                  tensor_uint8
 * Input T1 size:
 *   A scalar value indicating the length of the window.
 *   Allowed Types: tensor_int32, tensor_int64
 * Output T2 output:
 *   A Hamming window with length: size. The output has the shape: [size].
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_uint16, tensor_uint32, tensor_uint64,
 *                  tensor_uint8
 * Attribute INT output_datatype (optional):
 *   The data type of the output tensor. Strictly must be one of the values
 *   from DataType enum in TensorProto whose values correspond to T2. The
 *   default value is 1 = FLOAT.
 * 
 * Attribute INT periodic (optional):
 *   If 1, returns a window to be used as periodic function. If 0, return a
 *   symmetric window. When 'periodic' is specified, hann computes a window of
 *   length size + 1 and returns the first size points. The default value is 1.
 *
 * @since version 17
 *
 * @see github/workspace/onnx/defs/math/defs.cc:3190
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#HammingWindow
 */

operator_status
prepare_operator__ai_onnx__hammingwindow__17(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__hammingwindow__17;

typedef struct {
// no attributes
} context_operator__ai_onnx__hammingwindow__17;

operator_executer
resolve_operator__ai_onnx__hammingwindow__17(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__hammingwindow__17(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__hammingwindow__17__T1_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__hammingwindow__17__T1_tensor_int64(
    node_context *ctx
);

# endif