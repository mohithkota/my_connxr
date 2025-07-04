//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__maxunpool__9.h"
#include "tracing.h"
#include "utils.h"

void
free_operator__ai_onnx__maxunpool__9(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_I = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_output_shape = searchInputByName(ctx, 2);

    // TRACE_TENSOR(2, true, i_X);
    // TRACE_TENSOR(2, true, i_I);
    // TRACE_TENSOR(2, output_shape, i_output_shape);

    // Onnx__AttributeProto *a_kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"kernel_shape");
    // Onnx__AttributeProto *a_pads = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"pads");
    // Onnx__AttributeProto *a_strides = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"strides");

    // TRACE_ATTRIBUTE(2, true, a_kernel_shape);
    // TRACE_ATTRIBUTE(2, a_pads, a_pads);
    // TRACE_ATTRIBUTE(2, a_strides, a_strides);

    // Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_output);

    /* FREE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__maxunpool__9 *op_ctx = ctx->executer_context;

    TRACE_ARRAY(2, true, op_ctx->kernel_shape, , op_ctx->n_kernel_shape, "%" PRId64);
    TRACE_ARRAY(2, true, op_ctx->pads, , op_ctx->n_pads, "%" PRId64);
    TRACE_ARRAY(2, true, op_ctx->strides, , op_ctx->n_strides, "%" PRId64);

    

    // free(op_ctx);


    /* FREE OUTPUT DATA_TYPE AND SHAPE HERE */
    /* DO NOT FREE THE TENSOR ITSELF */

    // freeTensorData(o_output);
    // free(o_output->dims);

    TRACE_EXIT(1);
}