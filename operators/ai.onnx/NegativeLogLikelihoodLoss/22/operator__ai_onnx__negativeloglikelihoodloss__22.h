//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__NEGATIVELOGLIKELIHOODLOSS__22_H
# define OPERATOR_OPERATOR__AI_ONNX__NEGATIVELOGLIKELIHOODLOSS__22_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'NegativeLogLikelihoodLoss' version 22
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
 * Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
 * The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
 * The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
 * or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
 * The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:
 * 
 * ```
 * loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
 * ```
 * 
 * When an optional "weight" is provided, the sample loss is calculated as:
 * 
 * ```
 * loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
 * ```
 * 
 * loss is zero for the case when target-value equals ignore_index.
 * 
 * ```
 * loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
 * ```
 * 
 * If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
 * If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:
 * 
 * ```
 * mean(loss), if "weight" is not provided,
 * ```
 * 
 * or if weight is provided,
 * 
 * ```
 * sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
 * ```
 * 
 * If "reduction" attribute is set to "sum", the output is a scalar: `sum(loss)`.
 * 
 * See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.
 * 
 * Example 1:
 * 
 * ```
 * // negative log likelihood loss, "none" reduction
 * N, C, d1 = 2, 3, 2
 * input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
 *           [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
 * target = [[2, 1], [0, 2]]
 * 
 * loss = np.zeros((N, d1))
 * for n in range(N):
 *     for d_1 in range(d1):
 *         c = target[n][d_1]
 *         loss[n][d_1] = -input[n][c][d_1]
 * 
 * // print(loss)
 * // [[-3. -2.]
 * //  [-0. -2.]]
 * ```
 * 
 * Example 2:
 * 
 * ```
 * // weighted negative log likelihood loss, sum reduction
 * N, C, d1 = 2, 3, 2
 * input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
 *         [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
 * target = [[2, 1], [0, 2]]
 * weight = [0.2, 0.3, 0.1]
 * loss = np.zeros((N, d1))
 * for n in range(N):
 *     for d_1 in range(d1):
 *         c = target[n][d_1]
 *         loss[n][d_1] = -input[n][c][d_1] * weight[c]
 * 
 * loss = np.sum(loss)
 * // print(loss)
 * // -1.1
 * ```
 * 
 * Example 3:
 * 
 * ```
 * // weighted negative log likelihood loss, mean reduction
 * N, C, d1 = 2, 3, 2
 * input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
 *         [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
 * target = [[2, 1], [0, 2]]
 * weight = [0.2, 0.3, 0.1]
 * loss = np.zeros((N, d1))
 * weight_total = 0
 * for n in range(N):
 *     for d_1 in range(d1):
 *         c = target[n][d_1]
 *         loss[n][d_1] = -input[n][c][d_1] * weight[c]
 *         weight_total = weight_total + weight[c]
 * 
 * loss = np.sum(loss) / weight_total
 * // print(loss)
 * // -1.57
 * ```
 * 
 * Constraint T:
 *   Constrain input, weight, and output types to floating-point tensors.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Constraint Tind:
 *   Constrain target to integer types
 *   Allowed Types: tensor_int32, tensor_int64
 * Input T input:
 *   Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * 
 * Input Tind target:
 *   Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element value
 *   shall be in range of [0, C). If ignore_index is specified, it may have a
 *   value outside [0, C) and the target values should either be in the range
 *   [0, C) or have the value ignore_index.
 *   Allowed Types: tensor_int32, tensor_int64
 * 
 * Input T weight:
 *   Optional rescaling weight tensor. If given, it has to be a tensor of size
 *   C. Otherwise, it is treated as if having all ones.
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Output T loss:
 *   The negative log likelihood loss
 *   Allowed Types: tensor_bfloat16, tensor_double, tensor_float,
 *                  tensor_float16
 * Attribute INT ignore_index (optional):
 *   Specifies a target value that is ignored and does not contribute to the
 *   input gradient. It's an optional value.
 * 
 * Attribute STRING reduction (optional):
 *   Type of reduction to apply to loss: none, sum, mean (default). 'none':
 *   the output is the loss for each sample. 'sum': the output will be summed.
 *   'mean': the sum of the output will be divided by the sum of applied
 *   weights.
 *
 * @since version 22
 *
 * @see github/workspace/onnx/defs/math/defs.cc:2377
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#NegativeLogLikelihoodLoss
 */

operator_status
prepare_operator__ai_onnx__negativeloglikelihoodloss__22(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__negativeloglikelihoodloss__22;

typedef struct {
// no attributes
} context_operator__ai_onnx__negativeloglikelihoodloss__22;

operator_executer
resolve_operator__ai_onnx__negativeloglikelihoodloss__22(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_bfloat16__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_bfloat16__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_double__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_double__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_float__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_float__Tind_tensor_int64(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_float16__Tind_tensor_int32(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__negativeloglikelihoodloss__22__T_tensor_float16__Tind_tensor_int64(
    node_context *ctx
);

# endif