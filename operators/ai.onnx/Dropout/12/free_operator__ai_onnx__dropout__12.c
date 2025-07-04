//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__dropout__12.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__dropout__12(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_ratio = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_training_mode = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_data);
    // TRACE_TENSOR(2, ratio, i_ratio);
    // TRACE_TENSOR(2, training_mode, i_training_mode);

    // Onnx__AttributeProto *a_seed = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"seed");

    // TRACE_ATTRIBUTE(2, a_seed, a_seed);

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    // Onnx__TensorProto *o_mask = searchOutputByName(ctx, 1);

    // TRACE_TENSOR(2, true, o_output);
    // TRACE_TENSOR(2, mask, o_mask);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__dropout__12 *op_ctx = ctx->executer_context;

    TRACE_VAR(2, true, op_ctx->seed, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_output);
    // free(o_output->dims);
    // freeTensorData(o_mask);
    // free(o_mask->dims);

    TRACE_EXIT(1);
}