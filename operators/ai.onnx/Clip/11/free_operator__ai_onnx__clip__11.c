//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__clip__11.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__clip__11(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_min = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_max = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_input);
    // TRACE_TENSOR(2, min, i_min);
    // TRACE_TENSOR(2, max, i_max);

    

    

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_output);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__clip__11 *op_ctx = ctx->executer_context;

    

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_output);
    // free(o_output->dims);

    TRACE_EXIT(1);
}