//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__quantizelinear__19.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__quantizelinear__19(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_x = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_y_scale = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_y_zero_point = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_x);
    // TRACE_TENSOR(2, true, i_y_scale);
    // TRACE_TENSOR(2, y_zero_point, i_y_zero_point);

    // Onnx__AttributeProto *a_axis = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"axis");
    // Onnx__AttributeProto *a_saturate = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"saturate");

    // TRACE_ATTRIBUTE(2, a_axis, a_axis);
    // TRACE_ATTRIBUTE(2, a_saturate, a_saturate);

    // Onnx__TensorProto *o_y = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_y);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__quantizelinear__19 *op_ctx = ctx->executer_context;

    TRACE_VAR(2, true, op_ctx->axis, "%" PRId64);
    TRACE_VAR(2, true, op_ctx->saturate, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_y);
    // free(o_y->dims);

    TRACE_EXIT(1);
}