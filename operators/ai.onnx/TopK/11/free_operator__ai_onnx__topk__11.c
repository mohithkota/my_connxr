//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__topk__11.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__topk__11(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_K = searchInputByName(ctx, 1);

    // TRACE_TENSOR(2, true, i_X);
    // TRACE_TENSOR(2, true, i_K);

    // Onnx__AttributeProto *a_axis = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"axis");
    // Onnx__AttributeProto *a_largest = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"largest");
    // Onnx__AttributeProto *a_sorted = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"sorted");

    // TRACE_ATTRIBUTE(2, a_axis, a_axis);
    // TRACE_ATTRIBUTE(2, a_largest, a_largest);
    // TRACE_ATTRIBUTE(2, a_sorted, a_sorted);

    // Onnx__TensorProto *o_Values = searchOutputByName(ctx, 0);
    // Onnx__TensorProto *o_Indices = searchOutputByName(ctx, 1);

    // TRACE_TENSOR(2, true, o_Values);
    // TRACE_TENSOR(2, true, o_Indices);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__topk__11 *op_ctx = ctx->executer_context;

    TRACE_VAR(2, true, op_ctx->axis, "%" PRId64);
    TRACE_VAR(2, true, op_ctx->largest, "%" PRId64);
    TRACE_VAR(2, true, op_ctx->sorted, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_Values);
    // free(o_Values->dims);
    // freeTensorData(o_Indices);
    // free(o_Indices->dims);

    TRACE_EXIT(1);
}