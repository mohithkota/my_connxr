//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__negativeloglikelihoodloss__12.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__negativeloglikelihoodloss__12(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_target = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_weight = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_input);
    // TRACE_TENSOR(2, true, i_target);
    // TRACE_TENSOR(2, weight, i_weight);

    // Onnx__AttributeProto *a_ignore_index = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"ignore_index");
    // Onnx__AttributeProto *a_reduction = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"reduction");

    // TRACE_ATTRIBUTE(2, a_ignore_index, a_ignore_index);
    // TRACE_ATTRIBUTE(2, a_reduction, a_reduction);

    // Onnx__TensorProto *o_loss = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    

    // context_operator__ai_onnx__negativeloglikelihoodloss__12 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__negativeloglikelihoodloss__12));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    

    TRACE_VAR(2, true, op_ctx->ignore_index, "%" PRId64);
    TRACE_VAR(2, true, op_ctx->reduction, "\"%s\"");

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    // mallocTensorData(o_loss);

    // TRACE_TENSOR(2, true, o_loss);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    // ctx->executer = resolve_operator__ai_onnx__negativeloglikelihoodloss__12(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}