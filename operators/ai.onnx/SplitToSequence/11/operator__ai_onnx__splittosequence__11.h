//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__SPLITTOSEQUENCE__11_H
# define OPERATOR_OPERATOR__AI_ONNX__SPLITTOSEQUENCE__11_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'SplitToSequence' version 11
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Split a tensor into a sequence of tensors, along the specified 'axis'.
 * Lengths of the parts can be specified using the optional argument 'split'.
 * If the argument `split' is not specified, a default scalar value of 1
 * is used as the value of `split'.
 * 'split' must contain only positive numbers.
 * 'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
 * If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
 * if possible. The last chunk alone may be smaller than 'split' if the 'input' size
 * along the given axis 'axis' is not divisible by 'split'.
 * If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
 * with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
 * in 'split' must be equal to the dimension size of input tensor on 'axis'.
 * 
 * Constraint T:
 *   Constrain input types to all tensor types.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * 
 * Constraint I:
 *   Constrain split size to integral tensor.
 *   Allowed Types: tensor_int32, tensor_int64
 * 
 * Constraint S:
 *   Constrain output types to all tensor types.
 *   Allowed Types: seq_tensor_bool, seq_tensor_complex128,
 *                  seq_tensor_complex64, seq_tensor_double, seq_tensor_float,
 *                  seq_tensor_float16, seq_tensor_int16, seq_tensor_int32,
 *                  seq_tensor_int64, seq_tensor_int8, seq_tensor_string,
 *                  seq_tensor_uint16, seq_tensor_uint32, seq_tensor_uint64,
 *                  seq_tensor_uint8
 * Input T input:
 *   The tensor to split
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * 
 * Input I split:
 *   Length of each output. It can be either a scalar(tensor of empty shape),
 *   or a 1-D tensor. All values must be >= 0.
 *   Allowed Types: tensor_int32, tensor_int64
 * Output S output_sequence:
 *   One or more outputs forming a sequence of tensors after splitting
 *   Allowed Types: seq_tensor_bool, seq_tensor_complex128,
 *                  seq_tensor_complex64, seq_tensor_double, seq_tensor_float,
 *                  seq_tensor_float16, seq_tensor_int16, seq_tensor_int32,
 *                  seq_tensor_int64, seq_tensor_int8, seq_tensor_string,
 *                  seq_tensor_uint16, seq_tensor_uint32, seq_tensor_uint64,
 *                  seq_tensor_uint8
 * Attribute INT axis (optional):
 *   Which axis to split on. A negative value means counting dimensions from
 *   the back. Accepted range is [-rank, rank-1].
 * 
 * Attribute INT keepdims (optional):
 *   Keep the split dimension or not. Default 1, which means we keep split
 *   dimension. If input 'split' is specified, this attribute is ignored.
 *
 * @since version 11
 *
 * @see github/workspace/onnx/defs/sequence/defs.cc:274
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#SplitToSequence
 */

operator_status
prepare_operator__ai_onnx__splittosequence__11(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__splittosequence__11;

typedef struct {
// no attributes
} context_operator__ai_onnx__splittosequence__11;

operator_executer
resolve_operator__ai_onnx__splittosequence__11(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_bool__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_bool__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_complex128__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_complex128__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_complex64__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_complex64__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_double__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_double__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_float__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_float__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_float16__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_float16__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int16__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int16__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int32__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int32__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int64__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int64__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int8__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_int8__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_string__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_string__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint16__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint16__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint32__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint32__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint64__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint64__I_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint8__I_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__splittosequence__11__T_tensor_uint8__I_tensor_int64(
    node_context *ctx
);

# endif