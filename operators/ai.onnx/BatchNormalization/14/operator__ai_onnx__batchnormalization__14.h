//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__BATCHNORMALIZATION__14_H
# define OPERATOR_OPERATOR__AI_ONNX__BATCHNORMALIZATION__14_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'BatchNormalization' version 14
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Carries out batch normalization as described in the paper
 * https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
 * There are five required inputs 'X', 'scale', 'B', 'input_mean' and
 * 'input_var'.
 * Note that 'input_mean' and 'input_var' are expected to be the estimated
 * statistics in inference mode (training_mode=False, default),
 * and the running statistics in training mode (training_mode=True).
 * There are multiple cases for the number of outputs, which we list below:
 * 
 * Output case #1: Y, running_mean, running_var (training_mode=True)
 * Output case #2: Y (training_mode=False)
 * 
 * When training_mode=False, extra outputs are invalid.
 * The outputs are updated as follows when training_mode=True:
 * ```
 * running_mean = input_mean * momentum + current_mean * (1 - momentum)
 * running_var = input_var * momentum + current_var * (1 - momentum)
 * 
 * Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
 * 
 * where:
 * 
 * current_mean = ReduceMean(X, axis=all_except_channel_index)
 * current_var =  ReduceVar(X, axis=all_except_channel_index)
 * 
 * Notice that ReduceVar refers to the population variance, and it equals to
 * sum(sqrd(x_i - x_avg)) / N
 * where N is the population size (this formula does not use sample size N - 1).
 * 
 * ```
 * 
 * When training_mode=False:
 * ```
 * Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
 * ```
 * 
 * For previous (depreciated) non-spatial cases, implementors are suggested
 * to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
 * This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Constraint U:
 *   Constrain mean and variance types to float tensors. It allows all float
 *   type for U.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Input T X:
 *   Input data tensor from the previous operator; dimensions are in the form
 *   of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of
 *   channels. Statistics are computed for every channel of C over N and D1 to
 *   Dn dimensions. For image data, input dimensions become (N x C x H x W).
 *   The op also accepts single dimension input of size N in which case C is
 *   assumed to be 1
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Input T scale:
 *   Scale tensor of shape (C).
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Input T B:
 *   Bias tensor of shape (C).
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Input U input_mean:
 *   running (training) or estimated (testing) mean tensor of shape (C).
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Input U input_var:
 *   running (training) or estimated (testing) variance tensor of shape (C).
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Output T Y:
 *   The output tensor of the same shape as X
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Output U running_mean:
 *   The running mean after the BatchNormalization operator.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Output U running_var:
 *   The running variance after the BatchNormalization operator. This op uses
 *   the population size (N) for calculating variance, and not the sample size
 *   N-1.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Attribute FLOAT epsilon (optional):
 *   The epsilon value to use to avoid division by zero.
 * 
 * Attribute FLOAT momentum (optional):
 *   Factor used in computing the running mean and variance.e.g., running_mean
 *   = running_mean * momentum + mean * (1 - momentum).
 * 
 * Attribute INT training_mode (optional):
 *   If set to true, it indicates BatchNormalization is being used for
 *   training, and outputs 1, 2, 3, and 4 would be populated.
 *
 * @since version 14
 *
 * @see github/workspace/onnx/defs/nn/old.cc:3365
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
 */

operator_status
prepare_operator__ai_onnx__batchnormalization__14(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__batchnormalization__14;

typedef struct {
// no attributes
} context_operator__ai_onnx__batchnormalization__14;

operator_executer
resolve_operator__ai_onnx__batchnormalization__14(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_bfloat16__U_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_bfloat16__U_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_bfloat16__U_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_bfloat16__U_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_double__U_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_double__U_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_double__U_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_double__U_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float__U_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float__U_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float__U_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float__U_tensor_float16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float16__U_tensor_bfloat16(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float16__U_tensor_double(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float16__U_tensor_float(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__batchnormalization__14__T_tensor_float16__U_tensor_float16(
    node_context *ctx
);

# endif