//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__ISINF__10_H
# define OPERATOR_OPERATOR__AI_ONNX__ISINF__10_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'IsInf' version 10
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Map infinity to true and other values to false.
 * 
 * Constraint T1:
 *   Constrain input types to float tensors.
 *   Allowed Types: tensor_double, tensor_float
 * 
 * Constraint T2:
 *   Constrain output types to boolean tensors.
 *   Allowed Types: tensor_bool
 * Input T1 X:
 *   input
 *   Allowed Types: tensor_double, tensor_float
 * Output T2 Y:
 *   output
 *   Allowed Types: tensor_bool
 * Attribute INT detect_negative (optional):
 *   (Optional) Whether map negative infinity to true. Default to 1 so that
 *   negative infinity induces true. Set this attribute to 0 if negative
 *   infinity should be mapped to false.
 * 
 * Attribute INT detect_positive (optional):
 *   (Optional) Whether map positive infinity to true. Default to 1 so that
 *   positive infinity induces true. Set this attribute to 0 if positive
 *   infinity should be mapped to false.
 *
 * @since version 10
 *
 * @see github/workspace/onnx/defs/tensor/old.cc:3592
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#IsInf
 */

operator_status
prepare_operator__ai_onnx__isinf__10(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__isinf__10;

typedef struct {
// no attributes
} context_operator__ai_onnx__isinf__10;

operator_executer
resolve_operator__ai_onnx__isinf__10(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__isinf__10(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__isinf__10__T1_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__isinf__10__T1_tensor_float(
    node_context *ctx
);

# endif