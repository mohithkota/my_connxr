//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__LESSOREQUAL__12_H
# define OPERATOR_OPERATOR__AI_ONNX__LESSOREQUAL__12_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'LessOrEqual' version 12
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Returns the tensor resulted from performing the `less_equal` logical operation
 * elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
 * 
 * This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
 * 
 * Constraint T:
 *   Constrain input types to all numeric tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_uint16,
 *                  tensor_uint32, tensor_uint64, tensor_uint8
 * 
 * Constraint T1:
 *   Constrain output to boolean tensor.
 *   Allowed Types: tensor_bool
 * Input T A:
 *   First input operand for the logical operator.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_uint16,
 *                  tensor_uint32, tensor_uint64, tensor_uint8
 * 
 * Input T B:
 *   Second input operand for the logical operator.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_uint16,
 *                  tensor_uint32, tensor_uint64, tensor_uint8
 * Output T1 C:
 *   Result tensor.
 *   Allowed Types: tensor_bool

 *
 * @since version 12
 *
 * @see github/workspace/onnx/defs/logical/old.cc:219
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#LessOrEqual
 */

operator_status
prepare_operator__ai_onnx__lessorequal__12(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__lessorequal__12;

typedef struct {
// no attributes
} context_operator__ai_onnx__lessorequal__12;

operator_executer
resolve_operator__ai_onnx__lessorequal__12(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_int16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_uint16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_uint64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__lessorequal__12__T_tensor_uint8(
    node_context *ctx
);

# endif