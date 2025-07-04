//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__REVERSESEQUENCE__10_H
# define OPERATOR_OPERATOR__AI_ONNX__REVERSESEQUENCE__10_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'ReverseSequence' version 10
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Reverse batch of sequences having different lengths specified by `sequence_lens`.
 * 
 * For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
 * and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
 * sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.
 * 
 * Example 1:
 *   input = [[0.0, 4.0, 8.0,  12.0],
 *            [1.0, 5.0, 9.0,  13.0],
 *            [2.0, 6.0, 10.0, 14.0],
 *            [3.0, 7.0, 11.0, 15.0]]
 *   sequence_lens = [4, 3, 2, 1]
 *   time_axis = 0
 *   batch_axis = 1
 * 
 *   output = [[3.0, 6.0, 9.0,  12.0],
 *             [2.0, 5.0, 8.0,  13.0],
 *             [1.0, 4.0, 10.0, 14.0],
 *             [0.0, 7.0, 11.0, 15.0]]
 * 
 * Example 2:
 *   input = [[0.0,  1.0,  2.0,  3.0 ],
 *            [4.0,  5.0,  6.0,  7.0 ],
 *            [8.0,  9.0,  10.0, 11.0],
 *            [12.0, 13.0, 14.0, 15.0]]
 *   sequence_lens = [1, 2, 3, 4]
 *   time_axis = 1
 *   batch_axis = 0
 * 
 *   output = [[0.0,  1.0,  2.0,  3.0 ],
 *             [5.0,  4.0,  6.0,  7.0 ],
 *             [10.0, 9.0,  8.0,  11.0],
 *             [15.0, 14.0, 13.0, 12.0]]
 * 
 * Constraint T:
 *   Input and output types can be of any tensor type.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Input T input:
 *   Tensor of rank r >= 2.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * 
 * Input tensor(int64) sequence_lens:
 *   Tensor specifying lengths of the sequences in a batch. It has shape
 *   `[batch_size]`.
 *   Allowed Types: tensor_int64
 * Output T Y:
 *   Tensor with same shape of input.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Attribute INT batch_axis (optional):
 *   (Optional) Specify which axis is batch axis. Must be one of 1 (default),
 *   or 0.
 * 
 * Attribute INT time_axis (optional):
 *   (Optional) Specify which axis is time axis. Must be one of 0 (default),
 *   or 1.
 *
 * @since version 10
 *
 * @see github/workspace/onnx/defs/tensor/defs.cc:3125
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReverseSequence
 */

operator_status
prepare_operator__ai_onnx__reversesequence__10(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__reversesequence__10;

typedef struct {
// no attributes
} context_operator__ai_onnx__reversesequence__10;

operator_executer
resolve_operator__ai_onnx__reversesequence__10(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_complex128(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_complex64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_int16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_string(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_uint16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_uint64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reversesequence__10__T_tensor_uint8(
    node_context *ctx
);

# endif