//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__DROPOUT__12_H
# define OPERATOR_OPERATOR__AI_ONNX__DROPOUT__12_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Dropout' version 12
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
 * output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
 * Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
 * the user can simply not pass `training_mode` input or set it to false.
 * ```
 * output = scale * data * mask,
 * ```
 * where
 * ```
 * scale = 1. / (1. - ratio).
 * ```
 * This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Constraint T1:
 *   Constrain input 'ratio' types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Constraint T2:
 *   Constrain output 'mask' types to boolean tensors.
 *   Allowed Types: tensor_bool
 * Input T data:
 *   The input data as Tensor.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T1 ratio:
 *   The ratio of random dropout, with value in [0, 1). If this input was not
 *   set, or if it was set to 0, the output would be a simple copy of the
 *   input. If it's non-zero, output will be a random dropout of the scaled
 *   input, which is typically the case during training. It is an optional
 *   value, if not specified it will default to 0.5.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T2 training_mode:
 *   If set to true then it indicates dropout is being used for training. It
 *   is an optional value hence unless specified explicitly, it is false. If it
 *   is false, ratio is ignored and the operation mimics inference mode where
 *   nothing will be dropped from the input data and if mask is requested as
 *   output it will contain all ones.
 *   Allowed Types: tensor_bool
 * Output T output:
 *   The output.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Output T2 mask:
 *   The output mask.
 *   Allowed Types: tensor_bool
 * Attribute INT seed (optional):
 *   (Optional) Seed to the random generator, if not specified we will auto
 *   generate one.
 *
 * @since version 12
 *
 * @see github/workspace/onnx/defs/nn/old.cc:1560
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout
 */

operator_status
prepare_operator__ai_onnx__dropout__12(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__dropout__12;

typedef struct {
// no attributes
} context_operator__ai_onnx__dropout__12;

operator_executer
resolve_operator__ai_onnx__dropout__12(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_double__T1_tensor_double__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_double__T1_tensor_float__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_double__T1_tensor_float16__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_float__T1_tensor_double__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_float__T1_tensor_float__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_float__T1_tensor_float16__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_float16__T1_tensor_double__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_float16__T1_tensor_float__T2_tensor_bool(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__dropout__12__T_tensor_float16__T1_tensor_float16__T2_tensor_bool(
    node_context *ctx
);

# endif