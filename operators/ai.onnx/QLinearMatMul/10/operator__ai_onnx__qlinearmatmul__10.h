//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__QLINEARMATMUL__10_H
# define OPERATOR_OPERATOR__AI_ONNX__QLINEARMATMUL__10_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'QLinearMatMul' version 10
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
 * It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
 * and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
 * For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
 * Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
 * (per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
 * or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
 * an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
 * for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
 * have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
 * Production must never overflow, and accumulation may overflow if and only if in 32 bits.
 * 
 * Constraint T1:
 *   Constrain input a and its zero point data type to 8-bit integer tensor.
 *   Allowed Types: tensor_int8, tensor_uint8
 * 
 * Constraint T2:
 *   Constrain input b and its zero point data type to 8-bit integer tensor.
 *   Allowed Types: tensor_int8, tensor_uint8
 * 
 * Constraint T3:
 *   Constrain output y and its zero point data type to 8-bit integer tensor.
 *   Allowed Types: tensor_int8, tensor_uint8
 * Input T1 a:
 *   N-dimensional quantized matrix a
 *   Allowed Types: tensor_int8, tensor_uint8
 * 
 * Input tensor(float) a_scale:
 *   scale of quantized input a
 *   Allowed Types: tensor_float
 * 
 * Input T1 a_zero_point:
 *   zero point of quantized input a
 *   Allowed Types: tensor_int8, tensor_uint8
 * 
 * Input T2 b:
 *   N-dimensional quantized matrix b
 *   Allowed Types: tensor_int8, tensor_uint8
 * 
 * Input tensor(float) b_scale:
 *   scale of quantized input b
 *   Allowed Types: tensor_float
 * 
 * Input T2 b_zero_point:
 *   zero point of quantized input b
 *   Allowed Types: tensor_int8, tensor_uint8
 * 
 * Input tensor(float) y_scale:
 *   scale of quantized output y
 *   Allowed Types: tensor_float
 * 
 * Input T3 y_zero_point:
 *   zero point of quantized output y
 *   Allowed Types: tensor_int8, tensor_uint8
 * Output T3 y:
 *   Quantized matrix multiply results from a * b
 *   Allowed Types: tensor_int8, tensor_uint8

 *
 * @since version 10
 *
 * @see github/workspace/onnx/defs/math/old.cc:4097
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#QLinearMatMul
 */

operator_status
prepare_operator__ai_onnx__qlinearmatmul__10(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__qlinearmatmul__10;

typedef struct {
// no attributes
} context_operator__ai_onnx__qlinearmatmul__10;

operator_executer
resolve_operator__ai_onnx__qlinearmatmul__10(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_int8__T2_tensor_int8__T3_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_int8__T2_tensor_int8__T3_tensor_uint8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_int8__T2_tensor_uint8__T3_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_int8__T2_tensor_uint8__T3_tensor_uint8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_uint8__T2_tensor_int8__T3_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_uint8__T2_tensor_int8__T3_tensor_uint8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_uint8__T2_tensor_uint8__T3_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__qlinearmatmul__10__T1_tensor_uint8__T2_tensor_uint8__T3_tensor_uint8(
    node_context *ctx
);

# endif