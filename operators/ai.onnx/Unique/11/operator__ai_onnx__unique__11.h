//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__UNIQUE__11_H
# define OPERATOR_OPERATOR__AI_ONNX__UNIQUE__11_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Unique' version 11
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
 * Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.
 * 
 * This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
 * The first output tensor 'Y' contains all unique values or subtensors of the input.
 * The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
 * The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
 * The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.
 * 
 * Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.
 * 
 * https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
 * 
 * Example 1:
 * ```
 * input_X = [2, 1, 1, 3, 4, 3]
 * attribute_sorted = 0
 * attribute_axis = None
 * output_Y = [2, 1, 3, 4]
 * output_indices = [0, 1, 3, 4]
 * output_inverse_indices = [0, 1, 1, 2, 3, 2]
 * output_counts = [1, 2, 2, 1]
 * ```
 * 
 * Example 2:
 * ```
 * input_X = [[1, 3], [2, 3]]
 * attribute_sorted = 1
 * attribute_axis = None
 * output_Y = [1, 2, 3]
 * output_indices = [0, 2, 1]
 * output_inverse_indices = [0, 2, 1, 2]
 * output_counts = [1, 1, 2]
 * ```
 * 
 * Example 3:
 * ```
 * input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
 * attribute_sorted = 1
 * attribute_axis = 0
 * output_Y = [[1, 0, 0], [2, 3, 4]]
 * output_indices = [0, 2]
 * output_inverse_indices = [0, 0, 1]
 * output_counts = [2, 1]
 * ```
 * 
 * Example 4:
 * ```
 * input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
 *             [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
 * attribute_sorted = 1
 * attribute_axis = 1
 * ```
 * 
 * intermediate data are presented below for better understanding:
 * there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
 * ```
 * A: [[1, 1], [1, 1]],
 *    [[0, 1], [0, 1]],
 *    [[2, 1], [2, 1]],
 *    [[0, 1], [0, 1]].
 * ```
 * 
 * there are 3 unique subtensors:
 * ```
 * [[1, 1], [1, 1]],
 * [[0, 1], [0, 1]],
 * [[2, 1], [2, 1]].
 * ```
 * 
 * sorted unique subtensors:
 * ```
 * B: [[0, 1], [0, 1]],
 *    [[1, 1], [1, 1]],
 *    [[2, 1], [2, 1]].
 * ```
 * 
 * output_Y is constructed from B:
 * ```
 * [[[0. 1.], [1. 1.], [2. 1.]],
 *  [[0. 1.], [1. 1.], [2. 1.]]]
 * ```
 * 
 * output_indices is to map from B to A:
 * ```
 * [1, 0, 2]
 * ```
 * 
 * output_inverse_indices is to map from A to B:
 * ```
 * [1, 0, 2, 0]
 * ```
 * 
 * output_counts:
 * ```
 * [2, 1, 1]
 * ```
 * 
 * Constraint T:
 *   Input can be of any tensor type.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Input T X:
 *   A N-D input tensor that is to be processed.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Output T Y:
 *   A tensor of the same type as 'X' containing all the unique values or
 *   subtensors sliced along a provided 'axis' in 'X', either sorted or
 *   maintained in the same order they occur in input 'X'
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * 
 * Output tensor(int64) indices:
 *   A 1-D INT64 tensor containing indices of 'Y' elements' first occurrence
 *   in 'X'. When 'axis' is provided, it contains indices to subtensors in
 *   input 'X' on the 'axis'. When 'axis' is not provided, it contains indices
 *   to values in the flattened input tensor.
 *   Allowed Types: tensor_int64
 * 
 * Output tensor(int64) inverse_indices:
 *   A 1-D INT64 tensor containing, for elements of 'X', its corresponding
 *   indices in 'Y'. When 'axis' is provided, it contains indices to subtensors
 *   in output 'Y' on the 'axis'. When 'axis' is not provided, it contains
 *   indices to values in output 'Y'.
 *   Allowed Types: tensor_int64
 * 
 * Output tensor(int64) counts:
 *   A 1-D INT64 tensor containing the count of each element of 'Y' in input
 *   'X'
 *   Allowed Types: tensor_int64
 * Attribute INT axis (optional):
 *   (Optional) The dimension to apply unique. If not specified, the unique
 *   elements of the flattened input are returned. Negative value means
 *   counting dimensions from the back. Accepted range is [-r, r-1] where r =
 *   rank(input).
 * 
 * Attribute INT sorted (optional):
 *   (Optional) Whether to sort the unique elements in ascending order before
 *   returning as output. Must be one of 0, or 1 (default).
 *
 * @since version 11
 *
 * @see github/workspace/onnx/defs/tensor/defs.cc:3267
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unique
 */

operator_status
prepare_operator__ai_onnx__unique__11(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__unique__11;

typedef struct {
// no attributes
} context_operator__ai_onnx__unique__11;

operator_executer
resolve_operator__ai_onnx__unique__11(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_complex128(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_complex64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_int16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_int8(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_string(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_uint16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_uint32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_uint64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__unique__11__T_tensor_uint8(
    node_context *ctx
);

# endif