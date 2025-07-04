//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__INSTANCENORMALIZATION__6_H
# define OPERATOR_OPERATOR__AI_ONNX__INSTANCENORMALIZATION__6_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'InstanceNormalization' version 6
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Carries out instance normalization as described in the paper
 * https://arxiv.org/abs/1607.08022.
 * 
 * y = scale * (x - mean) / sqrt(variance + epsilon) + B,
 * where mean and variance are computed per instance per channel.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Input T input:
 *   Input data tensor from the previous operator; dimensions for image case
 *   are (N x C x H x W), where N is the batch size, C is the number of
 *   channels, and H and W are the height and the width of the data. For non
 *   image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
 *   where N is the batch size.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T scale:
 *   The input 1-dimensional scale tensor of size C.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T B:
 *   The input 1-dimensional bias tensor of size C.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T output:
 *   The output tensor of the same shape as input.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Attribute FLOAT epsilon (optional):
 *   The epsilon value to use to avoid division by zero.
 *
 * @since version 6
 *
 * @see github/workspace/onnx/defs/nn/old.cc:396
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization
 */

operator_status
prepare_operator__ai_onnx__instancenormalization__6(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__instancenormalization__6;

typedef struct {
// no attributes
} context_operator__ai_onnx__instancenormalization__6;

operator_executer
resolve_operator__ai_onnx__instancenormalization__6(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__instancenormalization__6(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__instancenormalization__6__T_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__instancenormalization__6__T_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__instancenormalization__6__T_tensor_float16(
    node_context *ctx
);

# endif