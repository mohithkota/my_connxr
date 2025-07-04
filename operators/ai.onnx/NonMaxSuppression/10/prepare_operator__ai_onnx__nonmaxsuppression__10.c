//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__nonmaxsuppression__10.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__nonmaxsuppression__10(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_boxes = searchInputByName(ctx, 0);
    // Onnx__TensorProto *i_scores = searchInputByName(ctx, 1);
    // Onnx__TensorProto *i_max_output_boxes_per_class = searchInputByName(ctx, 2);
    // Onnx__TensorProto *i_iou_threshold = searchInputByName(ctx, 3);
    // Onnx__TensorProto *i_score_threshold = searchInputByName(ctx, 4);

    // TRACE_TENSOR(2, true, i_boxes);
    // TRACE_TENSOR(2, true, i_scores);
    // TRACE_TENSOR(2, max_output_boxes_per_class, i_max_output_boxes_per_class);
    // TRACE_TENSOR(2, iou_threshold, i_iou_threshold);
    // TRACE_TENSOR(2, score_threshold, i_score_threshold);

    // Onnx__AttributeProto *a_center_point_box = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"center_point_box");

    // TRACE_ATTRIBUTE(2, a_center_point_box, a_center_point_box);

    // Onnx__TensorProto *o_selected_indices = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    

    // context_operator__ai_onnx__nonmaxsuppression__10 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__nonmaxsuppression__10));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    

    TRACE_VAR(2, true, op_ctx->center_point_box, "%" PRId64);

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    // mallocTensorData(o_selected_indices);

    // TRACE_TENSOR(2, true, o_selected_indices);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    // ctx->executer = resolve_operator__ai_onnx__nonmaxsuppression__10(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}