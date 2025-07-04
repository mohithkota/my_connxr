//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__DET__11_H
# define OPERATOR_OPERATOR__AI_ONNX__DET__11_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Det' version 11
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Det calculates determinant of a square matrix or batches of square matrices.
 * Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
 * and the inner-most 2 dimensions form square matrices.
 * The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
 * e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
 * 
 * Constraint T:
 *   Constrain input and output types to floating-point tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Input T X:
 *   Input tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T Y:
 *   Output tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16

 *
 * @since version 11
 *
 * @see github/workspace/onnx/defs/math/old.cc:347
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Det
 */

operator_status
prepare_operator__ai_onnx__det__11(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__det__11;

typedef struct {
// no attributes
} context_operator__ai_onnx__det__11;

operator_executer
resolve_operator__ai_onnx__det__11(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__det__11(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__det__11__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__det__11__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__det__11__T_tensor_float16(
    node_context *ctx
);

# endif