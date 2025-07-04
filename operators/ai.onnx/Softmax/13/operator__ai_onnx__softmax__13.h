//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__SOFTMAX__13_H
# define OPERATOR_OPERATOR__AI_ONNX__SOFTMAX__13_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Softmax' version 13
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * The operator computes the normalized exponential values for the given input:
 * 
 *  Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) 
 * 
 * The "axis" attribute indicates the dimension along which Softmax
 * will be performed. The output tensor has the same shape
 * and contains the Softmax values of the corresponding input.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Input T input:
 *   The input tensor of rank >= axis.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Output T output:
 *   The output values with the same shape as the input tensor.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Attribute INT axis (optional):
 *   Describes the dimension Softmax will be performed on. Negative value
 *   means counting dimensions from the back. Accepted range is [-r, r-1] where
 *   r = rank(input).
 *
 * @since version 13
 *
 * @see github/workspace/onnx/defs/math/defs.cc:1101
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
 */

operator_status
prepare_operator__ai_onnx__softmax__13(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__softmax__13;

typedef struct {
// no attributes
} context_operator__ai_onnx__softmax__13;

operator_executer
resolve_operator__ai_onnx__softmax__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__softmax__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__softmax__13__T_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__softmax__13__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__softmax__13__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__softmax__13__T_tensor_float16(
    node_context *ctx
);

# endif