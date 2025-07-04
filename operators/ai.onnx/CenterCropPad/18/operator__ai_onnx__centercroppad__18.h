//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__CENTERCROPPAD__18_H
# define OPERATOR_OPERATOR__AI_ONNX__CENTERCROPPAD__18_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'CenterCropPad' version 18
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Center crop or pad an input to given dimensions.
 * 
 * The crop/pad dimensions can be specified for a subset of the `axes`. Non-specified dimensions will not be
 * cropped or padded.
 * 
 * If the input dimensions are bigger than the crop shape, a centered cropping window is extracted from the input.
 * If the input dimensions are smaller than the crop shape, the input is padded on each side equally,
 * so that the input is centered in the output.
 * 
 * Constraint T:
 *   Constrain input and output types to all tensor types.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * 
 * Constraint Tind:
 *   Constrain indices to integer types
 *   Allowed Types: tensor_int32, tensor_int64
 * Input T input_data:
 *   Input to extract the centered crop from.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * 
 * Input Tind shape:
 *   1-D tensor representing the cropping window dimensions.
 *   Allowed Types: tensor_int32, tensor_int64
 * Output T output_data:
 *   Output data.
 *   Allowed Types: tensor_bfloat16, tensor_bool, tensor_complex128,
 *                  tensor_complex64, tensor_double, tensor_float,
 *                  tensor_float16, tensor_int16, tensor_int32, tensor_int64,
 *                  tensor_int8, tensor_string, tensor_uint16, tensor_uint32,
 *                  tensor_uint64, tensor_uint8
 * Attribute INTS axes (optional):
 *   If provided, it specifies a subset of axes that 'shape' refer to. If not
 *   provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data).
 *   Negative value means counting dimensions from the back. Accepted range is
 *   [-r, r-1], where r = rank(data). Behavior is undefined if an axis is
 *   repeated.
 *
 * @since version 18
 *
 * @see github/workspace/onnx/defs/tensor/defs.cc:3757
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#CenterCropPad
 */

operator_status
prepare_operator__ai_onnx__centercroppad__18(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__centercroppad__18;

typedef struct {
// no attributes
} context_operator__ai_onnx__centercroppad__18;

operator_executer
resolve_operator__ai_onnx__centercroppad__18(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_bfloat16__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_bfloat16__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_bool__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_bool__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_complex128__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_complex128__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_complex64__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_complex64__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_double__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_double__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_float__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_float__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_float16__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_float16__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int16__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int16__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int32__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int32__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int64__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int64__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int8__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_int8__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_string__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_string__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint16__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint16__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint32__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint32__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint64__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint64__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint8__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint8__Tind_tensor_int64(
    node_context *ctx
);

# endif