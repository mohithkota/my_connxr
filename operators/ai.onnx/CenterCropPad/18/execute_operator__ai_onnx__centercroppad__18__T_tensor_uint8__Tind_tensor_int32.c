//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__centercroppad__18.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__centercroppad__18__T_tensor_uint8__Tind_tensor_int32(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_input_data = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_shape = searchInputByName(ctx, 1);

    // TRACE_TENSOR(2, true, i_input_data);
    // TRACE_TENSOR(2, true, i_shape);

    // context_operator__ai_onnx__centercroppad__18 *op_ctx = ctx->executer_context;

    

    TRACE_ARRAY(2, true, axes, , n_axes, "%" PRId64);

    // Onnx__TensorProto *o_output_data = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_output_data);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}