//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__leakyrelu__1.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__leakyrelu__1(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    // TRACE_TENSOR(2, true, i_X);

    // Onnx__AttributeProto *a_alpha = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"alpha");
    // Onnx__AttributeProto *a_consumed_inputs = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"consumed_inputs");

    // TRACE_ATTRIBUTE(2, a_alpha, a_alpha);
    // TRACE_ATTRIBUTE(2, a_consumed_inputs, a_consumed_inputs);

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_Y);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__leakyrelu__1 *op_ctx = ctx->executer_context;

    TRACE_VAR(2, true, op_ctx->alpha, "%f");
    TRACE_ARRAY(2, true, op_ctx->consumed_inputs, , op_ctx->n_consumed_inputs, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_Y);
    // free(o_Y->dims);

    TRACE_EXIT(1);
}