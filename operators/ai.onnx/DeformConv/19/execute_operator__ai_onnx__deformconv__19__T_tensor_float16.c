//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__deformconv__19.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__deformconv__19__T_tensor_float16(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_W = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_offset = searchInputByName(ctx, 2);
    // Onnx__TensorProto *i_B = searchInputByName(ctx, 3);
    // Onnx__TensorProto *i_mask = searchInputByName(ctx, 4);

    // TRACE_TENSOR(2, true, i_X);
    // TRACE_TENSOR(2, true, i_W);
    // TRACE_TENSOR(2, true, i_offset);
    // TRACE_TENSOR(2, B, i_B);
    // TRACE_TENSOR(2, mask, i_mask);

    // context_operator__ai_onnx__deformconv__19 *op_ctx = ctx->executer_context;

    

    TRACE_ARRAY(2, true, dilations, , n_dilations, "%" PRId64);
    TRACE_VAR(2, true, group, "%" PRId64);
    TRACE_ARRAY(2, true, kernel_shape, , n_kernel_shape, "%" PRId64);
    TRACE_VAR(2, true, offset_group, "%" PRId64);
    TRACE_ARRAY(2, true, pads, , n_pads, "%" PRId64);
    TRACE_ARRAY(2, true, strides, , n_strides, "%" PRId64);

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_Y);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}