//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__sigmoid__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__sigmoid__1(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    // TRACE_TENSOR(2, true, i_X);

    // Onnx__AttributeProto *a_consumed_inputs = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"consumed_inputs");

    // TRACE_ATTRIBUTE(2, a_consumed_inputs, a_consumed_inputs);

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    

    // context_operator__ai_onnx__sigmoid__1 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__sigmoid__1));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    

    TRACE_ARRAY(2, true, op_ctx->consumed_inputs, , op_ctx->n_consumed_inputs, "%" PRId64);

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    // mallocTensorData(o_Y);

    // TRACE_TENSOR(2, true, o_Y);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    // ctx->executer = resolve_operator__ai_onnx__sigmoid__1(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}