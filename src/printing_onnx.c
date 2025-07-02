#include <stdio.h>
//#include <protobuf-c/protobuf-c.h>
#include "/home/mohithkota/myconnxr/protobuf/onnx.pb-c.h"
#include "/home/mohithkota/myconnxr/models/mnist/model_blob.h"

void main() {
    Onnx__ModelProto *model = onnx__model_proto__unpack(NULL, mohith_model_onnx_len, mohith_model_onnx);
    if (model == NULL) {
        printf("Failed to parse ONNX model\n");
        return;
    }

    printf("=== ONNX Model Info ===\n");
    printf("IR version: %lu\n", model->ir_version);
    printf("Producer: %s\n", model->producer_name);
    printf("Graph name: %s\n", model->graph->name);

    printf("\n-- Inputs --\n");
    for (size_t i = 0; i < model->graph->n_input; i++) {
        printf("Input %zu: %s\n", i, model->graph->input[i]->name);
    }

    printf("\n-- Outputs --\n");
    for (size_t i = 0; i < model->graph->n_output; i++) {
        printf("Output %zu: %s\n", i, model->graph->output[i]->name);
    }

    printf("\n-- Nodes (Operators) --\n");
    for (size_t i = 0; i < model->graph->n_node; i++) {
        printf("Node %zu: %s\n", i, model->graph->node[i]->op_type);
    }

    onnx__model_proto__free_unpacked(model, NULL);
}
