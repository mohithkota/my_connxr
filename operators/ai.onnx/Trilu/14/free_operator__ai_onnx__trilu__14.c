//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__trilu__14.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__trilu__14(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_k = searchInputByName(ctx, 1);

    // TRACE_TENSOR(2, true, i_input);
    // TRACE_TENSOR(2, k, i_k);

    // Onnx__AttributeProto *a_upper = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"upper");

    // TRACE_ATTRIBUTE(2, a_upper, a_upper);

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_output);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__trilu__14 *op_ctx = ctx->executer_context;

    TRACE_VAR(2, true, op_ctx->upper, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_output);
    // free(o_output->dims);

    TRACE_EXIT(1);
}