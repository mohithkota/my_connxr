//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__instancenormalization__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__instancenormalization__1__T_tensor_double(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_scale = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_B = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_input);
    // TRACE_TENSOR(2, true, i_scale);
    // TRACE_TENSOR(2, true, i_B);

    // context_operator__ai_onnx__instancenormalization__1 *op_ctx = ctx->executer_context;

    

    TRACE_ARRAY(2, true, consumed_inputs, , n_consumed_inputs, "%" PRId64);
    TRACE_VAR(2, true, epsilon, "%f");

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_output);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}