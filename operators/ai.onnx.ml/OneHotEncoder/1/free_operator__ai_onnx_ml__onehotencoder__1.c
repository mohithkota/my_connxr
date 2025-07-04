//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx_ml__onehotencoder__1.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx_ml__onehotencoder__1(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    // TRACE_TENSOR(2, true, i_X);

    // Onnx__AttributeProto *a_cats_int64s = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"cats_int64s");
    // Onnx__AttributeProto *a_cats_strings = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"cats_strings");
    // Onnx__AttributeProto *a_zeros = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"zeros");

    // TRACE_ATTRIBUTE(2, a_cats_int64s, a_cats_int64s);
    // TRACE_ATTRIBUTE(2, a_cats_strings, a_cats_strings);
    // TRACE_ATTRIBUTE(2, a_zeros, a_zeros);

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_Y);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx_ml__onehotencoder__1 *op_ctx = ctx->executer_context;

    TRACE_ARRAY(2, true, op_ctx->cats_int64s, , op_ctx->n_cats_int64s, "%" PRId64);
    TRACE_ARRAY(2, true, op_ctx->cats_strings, , op_ctx->n_cats_strings, "\"%s\"");
    TRACE_VAR(2, true, op_ctx->zeros, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_Y);
    // free(o_Y->dims);

    TRACE_EXIT(1);
}