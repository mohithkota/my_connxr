//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__dequantizelinear__10.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__dequantizelinear__10(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_x = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_x_scale = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_x_zero_point = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_x);
    // TRACE_TENSOR(2, true, i_x_scale);
    // TRACE_TENSOR(2, x_zero_point, i_x_zero_point);

    

    

    // Onnx__TensorProto *o_y = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_y);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__dequantizelinear__10 *op_ctx = ctx->executer_context;

    

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_y);
    // free(o_y->dims);

    TRACE_EXIT(1);
}