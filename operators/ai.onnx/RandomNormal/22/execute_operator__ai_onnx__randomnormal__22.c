//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__randomnormal__22.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__randomnormal__22(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    

    

    // context_operator__ai_onnx__randomnormal__22 *op_ctx = ctx->executer_context;

    

    TRACE_VAR(2, true, dtype, "%" PRId64);
    TRACE_VAR(2, true, mean, "%f");
    TRACE_VAR(2, true, scale, "%f");
    TRACE_VAR(2, true, seed, "%f");
    TRACE_ARRAY(2, true, shape, , n_shape, "%" PRId64);

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_output);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}