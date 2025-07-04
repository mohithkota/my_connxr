//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__WHERE__16_H
# define OPERATOR_OPERATOR__AI_ONNX__WHERE__16_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Where' version 16
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Return elements, either from X or Y, depending on condition.
 * Where behaves like
 * [numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
 * with three parameters.
 * 
 * This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
 * 
 * Constraint B:
 *   Constrain to boolean tensors.
 *   Allowed Types: tensor_bool
 * 
 * Constraint T:
 *   Constrain input and output types to all tensor types (including bfloat).
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Input B condition:
 *   When True (nonzero), yield X, otherwise yield Y
 *   Allowed Types: tensor_bool
 * 
 * Input T X:
 *   values selected at indices where condition is True
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * 
 * Input T Y:
 *   values selected at indices where condition is False
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Output T output:
 *   Tensor of shape equal to the broadcasted shape of condition, X, and Y.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8

 *
 * @since version 16
 *
 * @see github/workspace/onnx/defs/tensor/defs.cc:3011
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
 */

operator_status
prepare_operator__ai_onnx__where__16(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__where__16;

typedef struct {
// no attributes
} context_operator__ai_onnx__where__16;

operator_executer
resolve_operator__ai_onnx__where__16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_complex128(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_complex64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_int16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_string(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_uint16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_uint64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__where__16__B_tensor_bool__T_tensor_uint8(
    node_context *ctx
);

# endif