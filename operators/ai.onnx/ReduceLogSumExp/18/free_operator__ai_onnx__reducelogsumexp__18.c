//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__reducelogsumexp__18.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__reducelogsumexp__18(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_axes = searchInputByName(ctx, 1);

    // TRACE_TENSOR(2, true, i_data);
    // TRACE_TENSOR(2, axes, i_axes);

    // Onnx__AttributeProto *a_keepdims = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"keepdims");
    // Onnx__AttributeProto *a_noop_with_empty_axes = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"noop_with_empty_axes");

    // TRACE_ATTRIBUTE(2, a_keepdims, a_keepdims);
    // TRACE_ATTRIBUTE(2, a_noop_with_empty_axes, a_noop_with_empty_axes);

    // Onnx__TensorProto *o_reduced = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_reduced);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__reducelogsumexp__18 *op_ctx = ctx->executer_context;

    TRACE_VAR(2, true, op_ctx->keepdims, "%" PRId64);
    TRACE_VAR(2, true, op_ctx->noop_with_empty_axes, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_reduced);
    // free(o_reduced->dims);

    TRACE_EXIT(1);
}