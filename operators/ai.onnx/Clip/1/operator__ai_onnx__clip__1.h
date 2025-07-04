//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__CLIP__1_H
# define OPERATOR_OPERATOR__AI_ONNX__CLIP__1_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Clip' version 1
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Clip operator limits the given input within an interval. The interval is
 * specified with arguments 'min' and 'max'. They default to
 * numeric_limits::lowest() and numeric_limits::max() respectively.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Input T input:
 *   Input tensor whose elements to be clipped
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T output:
 *   Output tensor with clipped input elements
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Attribute INTS consumed_inputs (optional):
 *   legacy optimization attribute.
 * 
 * Attribute FLOAT max (optional):
 *   Maximum value, above which element is replaced by max
 * 
 * Attribute FLOAT min (optional):
 *   Minimum value, under which element is replaced by min
 *
 * @since version 1
 *
 * @see github/workspace/onnx/defs/math/old.cc:3213
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip
 */

operator_status
prepare_operator__ai_onnx__clip__1(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__clip__1;

typedef struct {
// no attributes
} context_operator__ai_onnx__clip__1;

operator_executer
resolve_operator__ai_onnx__clip__1(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__clip__1(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__clip__1__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__clip__1__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__clip__1__T_tensor_float16(
    node_context *ctx
);

# endif