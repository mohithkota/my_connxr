//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__NONMAXSUPPRESSION__11_H
# define OPERATOR_OPERATOR__AI_ONNX__NONMAXSUPPRESSION__11_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'NonMaxSuppression' version 11
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
 * Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
 * Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
 * orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
 * result in the same boxes being selected by the algorithm.
 * The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
 * The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
 * 

 * Input tensor(float) boxes:
 *   An input tensor with shape [num_batches, spatial_dimension, 4]. The
 *   single box data format is indicated by center_point_box.
 *   Allowed Types: tensor_float
 * 
 * Input tensor(float) scores:
 *   An input tensor with shape [num_batches, num_classes, spatial_dimension]
 *   Allowed Types: tensor_float
 * 
 * Input tensor(int64) max_output_boxes_per_class:
 *   Integer representing the maximum number of boxes to be selected per batch
 *   per class. It is a scalar. Default to 0, which means no output.
 *   Allowed Types: tensor_int64
 * 
 * Input tensor(float) iou_threshold:
 *   Float representing the threshold for deciding whether boxes overlap too
 *   much with respect to IOU. It is scalar. Value range [0, 1]. Default to 0.
 *   Allowed Types: tensor_float
 * 
 * Input tensor(float) score_threshold:
 *   Float representing the threshold for deciding when to remove boxes based
 *   on score. It is a scalar.
 *   Allowed Types: tensor_float
 * Output tensor(int64) selected_indices:
 *   selected indices from the boxes tensor. [num_selected_indices, 3], the
 *   selected index format is [batch_index, class_index, box_index].
 *   Allowed Types: tensor_int64
 * Attribute INT center_point_box (optional):
 *   Integer indicate the format of the box data. The default is 0. 0 - the
 *   box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are
 *   the coordinates of any diagonal pair of box corners and the coordinates
 *   can be provided as normalized (i.e., lying in the interval [0, 1]) or
 *   absolute. Mostly used for TF models. 1 - the box data is supplied as
 *   [x_center, y_center, width, height]. Mostly used for Pytorch models.
 *
 * @since version 11
 *
 * @see github/workspace/onnx/defs/object_detection/defs.cc:134
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression
 */

operator_status
prepare_operator__ai_onnx__nonmaxsuppression__11(
    node_context *ctx
);

extern operator_info info_operator__ai_onnx__nonmaxsuppression__11;

typedef struct {
// no attributes
} context_operator__ai_onnx__nonmaxsuppression__11;

operator_executer
resolve_operator__ai_onnx__nonmaxsuppression__11(
    node_context *ctx
);

operator_status
execute_operator__ai_onnx__nonmaxsuppression__11(
    node_context *ctx
);

# endif