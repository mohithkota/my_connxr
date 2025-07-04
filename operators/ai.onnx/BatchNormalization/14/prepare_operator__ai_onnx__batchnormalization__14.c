//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__batchnormalization__14.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__batchnormalization__14(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_scale = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_B = searchInputByName(ctx, 2);
    // Onnx__TensorProto *i_input_mean = searchInputByName(ctx, 3);
    // Onnx__TensorProto *i_input_var = searchInputByName(ctx, 4);

    // TRACE_TENSOR(2, true, i_X);
    // TRACE_TENSOR(2, true, i_scale);
    // TRACE_TENSOR(2, true, i_B);
    // TRACE_TENSOR(2, true, i_input_mean);
    // TRACE_TENSOR(2, true, i_input_var);

    // Onnx__AttributeProto *a_epsilon = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"epsilon");
    // Onnx__AttributeProto *a_momentum = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"momentum");
    // Onnx__AttributeProto *a_training_mode = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"training_mode");

    // TRACE_ATTRIBUTE(2, a_epsilon, a_epsilon);
    // TRACE_ATTRIBUTE(2, a_momentum, a_momentum);
    // TRACE_ATTRIBUTE(2, a_training_mode, a_training_mode);

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    // Onnx__TensorProto *o_running_mean = searchOutputByName(ctx, 1);
    // Onnx__TensorProto *o_running_var = searchOutputByName(ctx, 2);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    

    // context_operator__ai_onnx__batchnormalization__14 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__batchnormalization__14));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    

    TRACE_VAR(2, true, op_ctx->epsilon, "%f");
    TRACE_VAR(2, true, op_ctx->momentum, "%f");
    TRACE_VAR(2, true, op_ctx->training_mode, "%" PRId64);

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    // mallocTensorData(o_Y);
    // mallocTensorData(o_running_mean);
    // mallocTensorData(o_running_var);

    // TRACE_TENSOR(2, true, o_Y);
    // TRACE_TENSOR(2, running_mean, o_running_mean);
    // TRACE_TENSOR(2, running_var, o_running_var);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    // ctx->executer = resolve_operator__ai_onnx__batchnormalization__14(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}