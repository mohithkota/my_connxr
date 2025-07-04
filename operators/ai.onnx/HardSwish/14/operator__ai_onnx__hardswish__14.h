//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__HARDSWISH__14_H
# define OPERATOR_OPERATOR__AI_ONNX__HARDSWISH__14_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'HardSwish' version 14
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
 * the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
 * where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Input T X:
 *   Input tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T Y:
 *   Output tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16

 *
 * @since version 14
 *
 * @see github/workspace/onnx/defs/math/old.cc:778
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#HardSwish
 */

operator_status
prepare_operator__ai_onnx__hardswish__14(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__hardswish__14;

typedef struct {
// no attributes
} context_operator__ai_onnx__hardswish__14;

operator_executer
resolve_operator__ai_onnx__hardswish__14(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__hardswish__14(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__hardswish__14__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__hardswish__14__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__hardswish__14__T_tensor_float16(
    node_context *ctx
);

# endif