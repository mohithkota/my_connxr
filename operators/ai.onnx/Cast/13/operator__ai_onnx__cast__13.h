//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__CAST__13_H
# define OPERATOR_OPERATOR__AI_ONNX__CAST__13_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Cast' version 13
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * The operator casts the elements of a given input tensor to a data type
 * specified by the 'to' argument and returns an output tensor of the same size in
 * the converted type. The 'to' argument must be one of the data types specified
 * in the 'DataType' enum field in the TensorProto message.
 * 
 * Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
 * (e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
 * yield result 100. There are some string literals reserved for special floating-point values;
 * "+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
 * Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
 * this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
 * to string tensors, plain floating-point representation (such as "314.15926") would be used.
 * Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
 * of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.
 * 
 * Conversion from a numerical type to any numerical type is always allowed.
 * User must be aware of precision loss and value change caused by range difference between two types.
 * For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
 * an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.
 * 
 * In more detail, the conversion among numerical types should follow these rules:
 * 
 * * Casting from floating point to:
 *   * floating point: +/- infinity if OOR (out of range).
 *   * fixed point: undefined if OOR.
 *   * bool: +/- 0.0 to False; all else to True.
 * * Casting from fixed point to:
 *   * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
 *   * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
 *     signed types). For example, 200 (int16) -> -56 (int8).
 *   * bool: zero to False; nonzero to True.
 * * Casting from bool to:
 *   * floating point: `{1.0, 0.0}`.
 *   * fixed point: `{1, 0}`.
 *   * bool: no change.
 * 
 * Constraint T1:
 *   Constrain input types. Casting from complex is not supported.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * 
 * Constraint T2:
 *   Constrain output types. Casting to complex is not supported.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Input T1 input:
 *   Input tensor to be cast.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Output T2 output:
 *   Output tensor with the same shape as input with type specified by the
 *   'to' argument
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Attribute INT to :
 *   The data type to which the elements of the input tensor are cast.
 *   Strictly must be one of the types from DataType enum in TensorProto
 *
 * @since version 13
 *
 * @see github/workspace/onnx/defs/tensor/old.cc:300
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
 */

operator_status
prepare_operator__ai_onnx__cast__13(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__cast__13;

typedef struct {
// no attributes
} context_operator__ai_onnx__cast__13;

operator_executer
resolve_operator__ai_onnx__cast__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_int16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_string(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_uint16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_uint64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__cast__13__T1_tensor_uint8(
    node_context *ctx
);

# endif