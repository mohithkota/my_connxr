//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__QUANTIZELINEAR__19_H
# define OPERATOR_OPERATOR__AI_ONNX__QUANTIZELINEAR__19_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'QuantizeLinear' version 19
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
 * The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
 * The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
 * For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
 * For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
 * 'y_zero_point' and 'y' must have same type.
 * 'y_zero_point' is usually not used for quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz,
 * but the quantization formula remains the same for consistency and
 * the type of the attribute 'y_zero_point' still determines the quantization type.
 * 
 * Constraint T1:
 *   Constrain 'x' to float, float16, bfloat16 or int32 tensor.
 *   Allowed Types: tensor_bfloat16, tensor_float, tensor_float16, tensor_int32
 * 
 * Constraint T2:
 *   Constrain 'y_zero_point' and 'y' to 8-bit integer/float tensor.
 *   Allowed Types: tensor_tensor(float8e4m3fn), tensor_tensor(float8e4m3fnuz),
 *                  tensor_tensor(float8e5m2), tensor_tensor(float8e5m2fnuz),
 *                  tensor_int8, tensor_uint8
 * Input T1 x:
 *   N-D full precision Input tensor to be quantized.
 *   Allowed Types: tensor_bfloat16, tensor_float, tensor_float16, tensor_int32
 * 
 * Input T1 y_scale:
 *   Scale for doing quantization to get 'y'. It can be a scalar, which means
 *   per-tensor/layer quantization, or a 1-D Tensor for per-axis quantization.
 *   Allowed Types: tensor_bfloat16, tensor_float, tensor_float16, tensor_int32
 * 
 * Input T2 y_zero_point:
 *   Zero point for doing quantization to get 'y'. Shape must match y_scale.
 *   Default is uint8 with zero point of 0 if it's not specified.
 *   Allowed Types: tensor_tensor(float8e4m3fn), tensor_tensor(float8e4m3fnuz),
 *                  tensor_tensor(float8e5m2), tensor_tensor(float8e5m2fnuz),
 *                  tensor_int8, tensor_uint8
 * Output T2 y:
 *   N-D quantized output tensor. It has same shape as input 'x'.
 *   Allowed Types: tensor_tensor(float8e4m3fn), tensor_tensor(float8e4m3fnuz),
 *                  tensor_tensor(float8e5m2), tensor_tensor(float8e5m2fnuz),
 *                  tensor_int8, tensor_uint8
 * Attribute INT axis (optional):
 *   (Optional) The axis of the quantization dimension of the input tensor.
 *   Ignored for per-tensor quantization. Negative value means counting
 *   dimensions from the back. Accepted range is [-r, r-1] where r =
 *   rank(input).
 * 
 * Attribute INT saturate (optional):
 *   The parameter defines how the conversion behaves if an input value is out
 *   of range of the destination type. It only applies for float 8 quantization
 *   (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by
 *   default. All cases are fully described in two tables inserted in the
 *   operator description.
 *
 * @since version 19
 *
 * @see github/workspace/onnx/defs/quantization/old.cc:22
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#QuantizeLinear
 */

operator_status
prepare_operator__ai_onnx__quantizelinear__19(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__quantizelinear__19;

typedef struct {
// no attributes
} context_operator__ai_onnx__quantizelinear__19;

operator_executer
resolve_operator__ai_onnx__quantizelinear__19(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_bfloat16__T2_tensor_tensor(float8e4m3fn)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_bfloat16__T2_tensor_tensor(float8e4m3fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_bfloat16__T2_tensor_tensor(float8e5m2)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_bfloat16__T2_tensor_tensor(float8e5m2fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_bfloat16__T2_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_bfloat16__T2_tensor_uint8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float__T2_tensor_tensor(float8e4m3fn)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float__T2_tensor_tensor(float8e4m3fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float__T2_tensor_tensor(float8e5m2)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float__T2_tensor_tensor(float8e5m2fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float__T2_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float__T2_tensor_uint8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float16__T2_tensor_tensor(float8e4m3fn)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float16__T2_tensor_tensor(float8e4m3fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float16__T2_tensor_tensor(float8e5m2)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float16__T2_tensor_tensor(float8e5m2fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float16__T2_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_float16__T2_tensor_uint8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_int32__T2_tensor_tensor(float8e4m3fn)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_int32__T2_tensor_tensor(float8e4m3fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_int32__T2_tensor_tensor(float8e5m2)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_int32__T2_tensor_tensor(float8e5m2fnuz)(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_int32__T2_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__quantizelinear__19__T1_tensor_int32__T2_tensor_uint8(
    node_context *ctx
);

# endif