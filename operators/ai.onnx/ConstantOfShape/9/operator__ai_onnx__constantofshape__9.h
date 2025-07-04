//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__CONSTANTOFSHAPE__9_H
# define OPERATOR_OPERATOR__AI_ONNX__CONSTANTOFSHAPE__9_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'ConstantOfShape' version 9
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Generate a tensor with given value and shape.
 * 
 * Constraint T1:
 *   Constrain input types.
 *   Allowed Types: tensor_int64
 * 
 * Constraint T2:
 *   Constrain output types to be numerics.
 *   Allowed Types: tensor_bool, tensor_double, tensor_float, tensor_float16,
 *                  tensor_int16, tensor_int32, tensor_int64, tensor_int8,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Input T1 input:
 *   1D tensor. The shape of the expected output tensor. If empty tensor is
 *   given, the output would be a scalar. All values must be >= 0.
 *   Allowed Types: tensor_int64
 * Output T2 output:
 *   Output tensor of shape specified by 'input'.If attribute 'value' is
 *   specified, the value and datatype of the output tensor is taken from
 *   'value'.If attribute 'value' is not specified, the value in the output
 *   defaults to 0, and the datatype defaults to float32.
 *   Allowed Types: tensor_bool, tensor_double, tensor_float, tensor_float16,
 *                  tensor_int16, tensor_int32, tensor_int64, tensor_int8,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Attribute TENSOR value (optional):
 *   (Optional) The value of the output elements.Should be a one-element
 *   tensor. If not specified, it defaults to a tensor of value 0 and datatype
 *   float32
 *
 * @since version 9
 *
 * @see github/workspace/onnx/defs/generator/old.cc:713
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantOfShape
 */

operator_status
prepare_operator__ai_onnx__constantofshape__9(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__constantofshape__9;

typedef struct {
// no attributes
} context_operator__ai_onnx__constantofshape__9;

operator_executer
resolve_operator__ai_onnx__constantofshape__9(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__constantofshape__9(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__constantofshape__9__T1_tensor_int64(
    node_context *ctx
);

# endif