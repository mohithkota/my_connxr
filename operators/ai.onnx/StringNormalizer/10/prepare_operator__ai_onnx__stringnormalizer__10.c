//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__stringnormalizer__10.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__stringnormalizer__10(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    // TRACE_TENSOR(2, true, i_X);

    // Onnx__AttributeProto *a_case_change_action = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"case_change_action");
    // Onnx__AttributeProto *a_is_case_sensitive = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"is_case_sensitive");
    // Onnx__AttributeProto *a_locale = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"locale");
    // Onnx__AttributeProto *a_stopwords = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"stopwords");

    // TRACE_ATTRIBUTE(2, a_case_change_action, a_case_change_action);
    // TRACE_ATTRIBUTE(2, a_is_case_sensitive, a_is_case_sensitive);
    // TRACE_ATTRIBUTE(2, a_locale, a_locale);
    // TRACE_ATTRIBUTE(2, a_stopwords, a_stopwords);

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    

    // context_operator__ai_onnx__stringnormalizer__10 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__stringnormalizer__10));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    

    TRACE_VAR(2, true, op_ctx->case_change_action, "\"%s\"");
    TRACE_VAR(2, true, op_ctx->is_case_sensitive, "%" PRId64);
    TRACE_VAR(2, true, op_ctx->locale, "\"%s\"");
    TRACE_ARRAY(2, true, op_ctx->stopwords, , op_ctx->n_stopwords, "\"%s\"");

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    // mallocTensorData(o_Y);

    // TRACE_TENSOR(2, true, o_Y);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    // ctx->executer = resolve_operator__ai_onnx__stringnormalizer__10(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}