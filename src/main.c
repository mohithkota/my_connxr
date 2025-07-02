#include <stdio.h>
#include "onnx.pb-c.h"
#include "tensor_store.h"
#include "load_model.h"
#include "load_input.h"
#include "ops_runner.h"
#include "print_utils.h"
#include "mem.h" // For reset_pool

int main() {
    // Reset static memory pool
    reset_pool();

    // === Load and Inspect ONNX Model ===
    Onnx__ModelProto* model = load_model_blob();
    if (!model) {
        printf("❌ Failed to parse model\n");
        return 1;
    }

    printf("=== ONNX Model Info ===\n");
    printf("IR version: %u\n", model->ir_version);
    printf("Producer: %s\n", model->producer_name);
    printf("Graph name: %s\n\n", model->graph->name);

    // === Print Inputs ===
    printf("-- Inputs --\n");
    for (size_t i = 0; i < model->graph->n_input; i++) {
        printf("Input %zu: %s\n", i, model->graph->input[i]->name);
    }

    // === Print Outputs ===
    printf("\n-- Outputs --\n");
    for (size_t i = 0; i < model->graph->n_output; i++) {
        printf("Output %zu: %s\n", i, model->graph->output[i]->name);
    }

    // === Print Operators ===
    printf("\n-- Nodes (Operators) --\n");
    for (size_t i = 0; i < model->graph->n_node; i++) {
        printf("Node %zu: %s\n", i, model->graph->node[i]->op_type);
    }

    // === Load Input Tensor from .pb blob ===
    load_input_tensor_from_blob();  // loads and registers into tensor store

    // === Run Operators ===
    printf("\n=== Running Inference ===\n");
    run_model(model);  // node-by-node execution

    // === Print Final Output ===
    printf("\n=== Output Tensor ===\n");
    print_tensor("output");  // assuming output name is "output"

    return 0;
}

