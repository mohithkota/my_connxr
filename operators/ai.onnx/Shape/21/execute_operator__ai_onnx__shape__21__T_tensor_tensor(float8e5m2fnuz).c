//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__shape__21.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__shape__21__T_tensor_tensor(float8e5m2fnuz)(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_data = searchInputByName(ctx, 0);

    // TRACE_TENSOR(2, true, i_data);

    // context_operator__ai_onnx__shape__21 *op_ctx = ctx->executer_context;

    

    TRACE_VAR(2, true, end, "%" PRId64);
    TRACE_VAR(2, true, start, "%" PRId64);

    // Onnx__TensorProto *o_shape = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_shape);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}