//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__dft__20.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__dft__20(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_dft_length = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_axis = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_input);
    // TRACE_TENSOR(2, dft_length, i_dft_length);
    // TRACE_TENSOR(2, axis, i_axis);

    // Onnx__AttributeProto *a_inverse = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"inverse");
    // Onnx__AttributeProto *a_onesided = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"onesided");

    // TRACE_ATTRIBUTE(2, a_inverse, a_inverse);
    // TRACE_ATTRIBUTE(2, a_onesided, a_onesided);

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_output);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__dft__20 *op_ctx = ctx->executer_context;

    TRACE_VAR(2, true, op_ctx->inverse, "%" PRId64);
    TRACE_VAR(2, true, op_ctx->onesided, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_output);
    // free(o_output->dims);

    TRACE_EXIT(1);
}