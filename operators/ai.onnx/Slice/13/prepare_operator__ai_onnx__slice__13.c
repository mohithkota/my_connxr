//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__slice__13.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__slice__13(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_starts = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_ends = searchInputByName(ctx, 2);
    // Onnx__TensorProto *i_axes = searchInputByName(ctx, 3);
    // Onnx__TensorProto *i_steps = searchInputByName(ctx, 4);

    // TRACE_TENSOR(2, true, i_data);
    // TRACE_TENSOR(2, true, i_starts);
    // TRACE_TENSOR(2, true, i_ends);
    // TRACE_TENSOR(2, axes, i_axes);
    // TRACE_TENSOR(2, steps, i_steps);

    

    

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    

    // context_operator__ai_onnx__slice__13 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__slice__13));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    

    

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    // mallocTensorData(o_output);

    // TRACE_TENSOR(2, true, o_output);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    // ctx->executer = resolve_operator__ai_onnx__slice__13(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}