//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__reshape__5.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__reshape__5(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_shape = searchInputByName(ctx, 1);

    // TRACE_TENSOR(2, true, i_data);
    // TRACE_TENSOR(2, true, i_shape);

    

    

    // Onnx__TensorProto *o_reshaped = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_reshaped);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__reshape__5 *op_ctx = ctx->executer_context;

    

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_reshaped);
    // free(o_reshaped->dims);

    TRACE_EXIT(1);
}