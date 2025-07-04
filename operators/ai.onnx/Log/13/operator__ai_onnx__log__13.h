//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__LOG__13_H
# define OPERATOR_OPERATOR__AI_ONNX__LOG__13_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Log' version 13
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Calculates the natural log of the given input tensor, element-wise.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Input T input:
 *   Input tensor
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Output T output:
 *   The natural log of the input tensor computed element-wise
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16

 *
 * @since version 13
 *
 * @see github/workspace/onnx/defs/math/defs.cc:651
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
 */

operator_status
prepare_operator__ai_onnx__log__13(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__log__13;

typedef struct {
// no attributes
} context_operator__ai_onnx__log__13;

operator_executer
resolve_operator__ai_onnx__log__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__log__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__log__13__T_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__log__13__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__log__13__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__log__13__T_tensor_float16(
    node_context *ctx
);

# endif