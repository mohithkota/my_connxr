//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__ELU__22_H
# define OPERATOR_OPERATOR__AI_ONNX__ELU__22_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Elu' version 22
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Elu takes one input data (Tensor<T>) and produces one output data
 * (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
 * 0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Input T X:
 *   1D input tensor
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Output T Y:
 *   1D output tensor
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Attribute FLOAT alpha (optional):
 *   Coefficient of ELU.
 *
 * @since version 22
 *
 * @see github/workspace/onnx/defs/math/defs.cc:417
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Elu
 */

operator_status
prepare_operator__ai_onnx__elu__22(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__elu__22;

typedef struct {
// no attributes
} context_operator__ai_onnx__elu__22;

operator_executer
resolve_operator__ai_onnx__elu__22(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__elu__22(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__elu__22__T_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__elu__22__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__elu__22__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__elu__22__T_tensor_float16(
    node_context *ctx
);

# endif