//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__SPLIT__18_H
# define OPERATOR_OPERATOR__AI_ONNX__SPLIT__18_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Split' version 18
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Split a tensor into a list of tensors, along the specified 'axis'.
 * Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
 * If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
 * If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
 * If the input 'split' is specified, it indicates the sizes of each output in the split.
 * 
 * Constraint T:
 *   Constrain input and output types to all tensor types.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Input T input:
 *   The tensor to split
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * 
 * Input tensor(int64) split:
 *   Optional length of each output. Values should be >= 0.Sum of the values
 *   must be equal to the dim value at 'axis' specified.
 *   Allowed Types: tensor_int64
 * Output T outputs:
 *   One or more outputs forming list of tensors after splitting
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Attribute INT axis (optional):
 *   Which axis to split on. A negative value means counting dimensions from
 *   the back. Accepted range is [-rank, rank-1] where r = rank(input).
 * 
 * Attribute INT num_outputs (optional):
 *   Number of outputs to split parts of the tensor into. If the tensor is not
 *   evenly splittable the last chunk will be smaller.
 *
 * @since version 18
 *
 * @see github/workspace/onnx/defs/tensor/defs.cc:618
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split
 */

operator_status
prepare_operator__ai_onnx__split__18(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__split__18;

typedef struct {
// no attributes
} context_operator__ai_onnx__split__18;

operator_executer
resolve_operator__ai_onnx__split__18(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_complex128(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_complex64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_int16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_string(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_uint16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_uint64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__split__18__T_tensor_uint8(
    node_context *ctx
);

# endif