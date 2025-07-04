//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__DEQUANTIZELINEAR__19_H
# define OPERATOR_OPERATOR__AI_ONNX__DEQUANTIZELINEAR__19_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'DequantizeLinear' version 19
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
 * The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
 * for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
 * `x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
 * there's no zero point (zero point is supposed to be 0).
 * `zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
 * but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.
 * 
 * Constraint T1:
 *   Constrain 'x_zero_point' and 'x' to 8-bit integer or float, or /32-bit
 *   integer tensor.
 *   Allowed Types: tensor_tensor(float8e4m3fn), tensor_tensor(float8e4m3fnuz),
 *                  tensor_tensor(float8e5m2), tensor_tensor(float8e5m2fnuz),
 *                  tensor_int32, tensor_int8, tensor_uint8
 * 
 * Constraint T2:
 *   'x_scale' determines the output type.
 *   Allowed Types: tensor_bfloat16, tensor_float, tensor_float16
 * Input T1 x:
 *   N-D quantized input tensor to be de-quantized.
 *   Allowed Types: tensor_tensor(float8e4m3fn), tensor_tensor(float8e4m3fnuz),
 *                  tensor_tensor(float8e5m2), tensor_tensor(float8e5m2fnuz),
 *                  tensor_int32, tensor_int8, tensor_uint8
 * 
 * Input T2 x_scale:
 *   Scale for input 'x'. It can be a scalar, which means a per-tensor/layer
 *   dequantization, or a 1-D tensor for per-axis dequantization.
 *   Allowed Types: tensor_bfloat16, tensor_float, tensor_float16
 * 
 * Input T1 x_zero_point:
 *   Zero point for input 'x'. Shape must match x_scale. It's optional. Zero
 *   point is 0 when it's not specified.
 *   Allowed Types: tensor_tensor(float8e4m3fn), tensor_tensor(float8e4m3fnuz),
 *                  tensor_tensor(float8e5m2), tensor_tensor(float8e5m2fnuz),
 *                  tensor_int32, tensor_int8, tensor_uint8
 * Output T2 y:
 *   N-D full precision output tensor. It has same shape as input 'x'.
 *   Allowed Types: tensor_bfloat16, tensor_float, tensor_float16
 * Attribute INT axis (optional):
 *   (Optional) The axis of the dequantizing dimension of the input tensor.
 *   Used only for per-axis quantization. Negative value means counting
 *   dimensions from the back. Accepted range is `[-r, r-1]` where `r =
 *   rank(input)`. When the rank of the input is 1, per-tensor quantization is
 *   applied, rendering the axis unnecessary in this scenario.
 *
 * @since version 19
 *
 * @see github/workspace/onnx/defs/quantization/old.cc:92
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#DequantizeLinear
 */

operator_status
prepare_operator__ai_onnx__dequantizelinear__19(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__dequantizelinear__19;

typedef struct {
// no attributes
} context_operator__ai_onnx__dequantizelinear__19;

operator_executer
resolve_operator__ai_onnx__dequantizelinear__19(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e4m3fn)__T2_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e4m3fn)__T2_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e4m3fn)__T2_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e4m3fnuz)__T2_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e4m3fnuz)__T2_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e4m3fnuz)__T2_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e5m2)__T2_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e5m2)__T2_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e5m2)__T2_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e5m2fnuz)__T2_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e5m2fnuz)__T2_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_tensor(float8e5m2fnuz)__T2_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_int32__T2_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_int32__T2_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_int32__T2_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_int8__T2_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_int8__T2_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_int8__T2_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_uint8__T2_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_uint8__T2_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dequantizelinear__19__T1_tensor_uint8__T2_tensor_float16(
    node_context *ctx
);

# endif