//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx_ml__scaler__1.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx_ml__scaler__1(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    // TRACE_TENSOR(2, true, i_X);

    // Onnx__AttributeProto *a_offset = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"offset");
    // Onnx__AttributeProto *a_scale = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"scale");

    // TRACE_ATTRIBUTE(2, a_offset, a_offset);
    // TRACE_ATTRIBUTE(2, a_scale, a_scale);

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_Y);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx_ml__scaler__1 *op_ctx = ctx->executer_context;

    TRACE_ARRAY(2, true, op_ctx->offset, , op_ctx->n_offset, "%f");
    TRACE_ARRAY(2, true, op_ctx->scale, , op_ctx->n_scale, "%f");

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_Y);
    // free(o_Y->dims);

    TRACE_EXIT(1);
}