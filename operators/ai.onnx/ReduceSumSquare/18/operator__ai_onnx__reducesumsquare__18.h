//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__REDUCESUMSQUARE__18_H
# define OPERATOR_OPERATOR__AI_ONNX__REDUCESUMSQUARE__18_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'ReduceSumSquare' version 18
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Computes the sum square of the input tensor's elements along the provided axes. The resulting
 * tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
 * the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
 * valid. Reduction over an empty set of values yields 0.
 * 
 * 
 * The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
 * to `False` instead of `True`.
 * 
 * Constraint T:
 *   Constrain input and output types to numeric tensors.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int32, tensor_int64, tensor_uint32,
 *                  tensor_uint64
 * Input T data:
 *   An input tensor.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int32, tensor_int64, tensor_uint32,
 *                  tensor_uint64
 * 
 * Input tensor(int64) axes:
 *   Optional input list of integers, along which to reduce. The default is to
 *   reduce over all the dimensions of the input tensor if
 *   'noop_with_empty_axes' is false, else act as an Identity op when
 *   'noop_with_empty_axes' is true. Accepted range is [-r, r-1] where r =
 *   rank(data).
 *   Allowed Types: tensor_int64
 * Output T reduced:
 *   Reduced output tensor.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int32, tensor_int64, tensor_uint32,
 *                  tensor_uint64
 * Attribute INT keepdims (optional):
 *   Keep the reduced dimension or not, default 1 means keep reduced
 *   dimension.
 * 
 * Attribute INT noop_with_empty_axes (optional):
 *   Defines behavior if 'axes' is empty. Default behavior with 'false' is to
 *   reduce all axes. When axes is empty and this attribute is set to true,
 *   input tensor will not be reduced,and the output tensor would be equivalent
 *   to input tensor.
 *
 * @since version 18
 *
 * @see github/workspace/onnx/defs/reduction/defs.cc:34
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSumSquare
 */

operator_status
prepare_operator__ai_onnx__reducesumsquare__18(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__reducesumsquare__18;

typedef struct {
// no attributes
} context_operator__ai_onnx__reducesumsquare__18;

operator_executer
resolve_operator__ai_onnx__reducesumsquare__18(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__reducesumsquare__18__T_tensor_uint64(
    node_context *ctx
);

# endif