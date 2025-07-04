//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__STRINGCONCAT__20_H
# define OPERATOR_OPERATOR__AI_ONNX__STRINGCONCAT__20_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'StringConcat' version 20
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support)
 * 
 * Constraint T:
 *   Inputs and outputs must be UTF-8 strings
 *   Allowed Types: tensor_string
 * Input T X:
 *   Tensor to prepend in concatenation
 *   Allowed Types: tensor_string
 * 
 * Input T Y:
 *   Tensor to append in concatenation
 *   Allowed Types: tensor_string
 * Output T Z:
 *   Concatenated string tensor
 *   Allowed Types: tensor_string

 *
 * @since version 20
 *
 * @see github/workspace/onnx/defs/text/defs.cc:10
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#StringConcat
 */

operator_status
prepare_operator__ai_onnx__stringconcat__20(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__stringconcat__20;

typedef struct {
// no attributes
} context_operator__ai_onnx__stringconcat__20;

operator_executer
resolve_operator__ai_onnx__stringconcat__20(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__stringconcat__20(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__stringconcat__20__T_tensor_string(
    node_context *ctx
);

# endif