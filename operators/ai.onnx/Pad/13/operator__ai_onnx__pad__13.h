//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__PAD__13_H
# define OPERATOR_OPERATOR__AI_ONNX__PAD__13_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Pad' version 13
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
 * a padded tensor (`output`) is generated.
 * 
 * The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):
 * 
 * 1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)
 * 
 * 2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
 * 
 * 3) `edge` - pads with the edge values of array
 * 
 * 
 * Example 1 (`constant` mode):
 *   Insert 0 pads to the beginning of the second dimension.
 * 
 *   data =
 *   [
 *       [1.0, 1.2],
 *       [2.3, 3.4],
 *       [4.5, 5.7],
 *   ]
 * 
 *   pads = [0, 2, 0, 0]
 * 
 *   mode = 'constant'
 * 
 *   constant_value = 0.0
 * 
 *   output =
 *   [
 *       [0.0, 0.0, 1.0, 1.2],
 *       [0.0, 0.0, 2.3, 3.4],
 *       [0.0, 0.0, 4.5, 5.7],
 *   ]
 * 
 * 
 * Example 2 (`reflect` mode):
 *   data =
 *   [
 *       [1.0, 1.2],
 *       [2.3, 3.4],
 *       [4.5, 5.7],
 *   ]
 * 
 *   pads = [0, 2, 0, 0]
 * 
 *   mode = 'reflect'
 * 
 *   output =
 *   [
 *       [1.0, 1.2, 1.0, 1.2],
 *       [2.3, 3.4, 2.3, 3.4],
 *       [4.5, 5.7, 4.5, 5.7],
 *   ]
 * 
 * 
 * Example 3 (`edge` mode):
 *   data =
 *   [
 *       [1.0, 1.2],
 *       [2.3, 3.4],
 *       [4.5, 5.7],
 *   ]
 * 
 *   pads = [0, 2, 0, 0]
 * 
 *   mode = 'edge'
 * 
 *   output =
 *   [
 *       [1.0, 1.0, 1.0, 1.2],
 *       [2.3, 2.3, 2.3, 3.4],
 *       [4.5, 4.5, 4.5, 5.7],
 *   ]
 * 
 * Constraint T:
 *   Constrain input and output types to all tensor types.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Input T data:
 *   Input tensor.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * 
 * Input tensor(int64) pads:
 *   Tensor of integers indicating the number of padding elements to add or
 *   remove (if negative) at the beginning and end of each axis. For 2D input
 *   tensor, it is the number of pixels. `pads` should be a 1D tensor of shape
 *   [2 * input_rank]. `pads` format should be: [x1_begin, x2_begin,...,x1_end,
 *   x2_end,...], where xi_begin is the number of pad values added at the
 *   beginning of axis `i` and xi_end, the number of pad values added at the
 *   end of axis `i`.
 *   Allowed Types: tensor_int64
 * 
 * Input T constant_value:
 *   (Optional) A scalar value to be used if the mode chosen is `constant` (by
 *   default it is 0, empty string or False).
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Output T output:
 *   Tensor after padding.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Attribute STRING mode (optional):
 *   Supported modes: `constant`(default), `reflect`, `edge`
 *
 * @since version 13
 *
 * @see github/workspace/onnx/defs/tensor/old.cc:5792
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad
 */

operator_status
prepare_operator__ai_onnx__pad__13(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__pad__13;

typedef struct {
// no attributes
} context_operator__ai_onnx__pad__13;

operator_executer
resolve_operator__ai_onnx__pad__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_complex128(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_complex64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_int16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_string(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_uint16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_uint64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__pad__13__T_tensor_uint8(
    node_context *ctx
);

# endif