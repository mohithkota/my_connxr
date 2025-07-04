//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__QUANTIZELINEAR__13_H
# define OPERATOR_OPERATOR__AI_ONNX__QUANTIZELINEAR__13_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'QuantizeLinear' version 13
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
 * The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
 * The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
 * For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
 * For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
 * 
 * Constraint T1:
 *   Constrain 'x' to float or int32 tensor.
 *   Allowed Types: tensor_float, tensor_int32
 * 
 * Constraint T2:
 *   Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.
 *   Allowed Types: tensor_int8, tensor_uint8
 * Input T1 x:
 *   N-D full precision Input tensor to be quantized.
 *   Allowed Types: tensor_float, tensor_int32
 * 
 * Input tensor(float) y_scale:
 *   Scale for doing quantization to get 'y'. It can be a scalar, which means
 *   per-tensor/layer quantization, or a 1-D Tensor for per-axis quantization.
 *   Allowed Types: tensor_float
 * 
 * Input T2 y_zero_point:
 *   Zero point for doing quantization to get 'y'. Shape must match y_scale.
 *   Default is uint8 with zero point of 0 if it's not specified.
 *   Allowed Types: tensor_int8, tensor_uint8
 * Output T2 y:
 *   N-D quantized output tensor. It has same shape as input 'x'.
 *   Allowed Types: tensor_int8, tensor_uint8
 * Attribute INT axis (optional):
 *   (Optional) The axis of the quantization dimension of the input tensor.
 *   Ignored for per-tensor quantization. Negative value means counting
 *   dimensions from the back. Accepted range is [-r, r-1] where r =
 *   rank(input).
 *
 * @since version 13
 *
 * @see github/workspace/onnx/defs/quantization/old.cc:151
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#QuantizeLinear
 */

operator_status
prepare_operator__ai_onnx__quantizelinear__13(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__quantizelinear__13;

typedef struct {
// no attributes
} context_operator__ai_onnx__quantizelinear__13;

operator_executer
resolve_operator__ai_onnx__quantizelinear__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__13__T1_tensor_float__T2_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__13__T1_tensor_float__T2_tensor_uint8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__13__T1_tensor_int32__T2_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__13__T1_tensor_int32__T2_tensor_uint8(
    node_context *ctx
);

# endif