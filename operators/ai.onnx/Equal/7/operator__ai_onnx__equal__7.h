//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__EQUAL__7_H
# define OPERATOR_OPERATOR__AI_ONNX__EQUAL__7_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Equal' version 7
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Returns the tensor resulted from performing the `equal` logical operation
 * elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
 * 
 * This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
 * 
 * Constraint T:
 *   Constrain input to integral tensors.
 *   Allowed Types: tensor_bool, tensor_int32, tensor_int64
 * 
 * Constraint T1:
 *   Constrain output to boolean tensor.
 *   Allowed Types: tensor_bool
 * Input T A:
 *   First input operand for the logical operator.
 *   Allowed Types: tensor_bool, tensor_int32, tensor_int64
 * 
 * Input T B:
 *   Second input operand for the logical operator.
 *   Allowed Types: tensor_bool, tensor_int32, tensor_int64
 * Output T1 C:
 *   Result tensor.
 *   Allowed Types: tensor_bool

 *
 * @since version 7
 *
 * @see github/workspace/onnx/defs/logical/old.cc:186
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
 */

operator_status
prepare_operator__ai_onnx__equal__7(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__equal__7;

typedef struct {
// no attributes
} context_operator__ai_onnx__equal__7;

operator_executer
resolve_operator__ai_onnx__equal__7(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__equal__7(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__equal__7__T_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__equal__7__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__equal__7__T_tensor_int64(
    node_context *ctx
);

# endif