//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx_ml__treeensembleregressor__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx_ml__treeensembleregressor__1(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    // Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    // TRACE_TENSOR(2, true, i_X);

    // context_operator__ai_onnx_ml__treeensembleregressor__1 *op_ctx = ctx->executer_context;

    

    TRACE_VAR(2, true, aggregate_function, "\"%s\"");
    TRACE_ARRAY(2, true, base_values, , n_base_values, "%f");
    TRACE_VAR(2, true, n_targets, "%" PRId64);
    TRACE_ARRAY(2, true, nodes_falsenodeids, , n_nodes_falsenodeids, "%" PRId64);
    TRACE_ARRAY(2, true, nodes_featureids, , n_nodes_featureids, "%" PRId64);
    TRACE_ARRAY(2, true, nodes_hitrates, , n_nodes_hitrates, "%f");
    TRACE_ARRAY(2, true, nodes_missing_value_tracks_true, , n_nodes_missing_value_tracks_true, "%" PRId64);
    TRACE_ARRAY(2, true, nodes_modes, , n_nodes_modes, "\"%s\"");
    TRACE_ARRAY(2, true, nodes_nodeids, , n_nodes_nodeids, "%" PRId64);
    TRACE_ARRAY(2, true, nodes_treeids, , n_nodes_treeids, "%" PRId64);
    TRACE_ARRAY(2, true, nodes_truenodeids, , n_nodes_truenodeids, "%" PRId64);
    TRACE_ARRAY(2, true, nodes_values, , n_nodes_values, "%f");
    TRACE_VAR(2, true, post_transform, "\"%s\"");
    TRACE_ARRAY(2, true, target_ids, , n_target_ids, "%" PRId64);
    TRACE_ARRAY(2, true, target_nodeids, , n_target_nodeids, "%" PRId64);
    TRACE_ARRAY(2, true, target_treeids, , n_target_treeids, "%" PRId64);
    TRACE_ARRAY(2, true, target_weights, , n_target_weights, "%f");

    // Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    // TRACE_TENSOR(2, true, o_Y);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}