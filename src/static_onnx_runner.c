
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "onnx.pb-c.h"
#include "model_blob.h"  // Contains: unsigned char model_blob[] and unsigned int model_blob_len

// === Static Tensor Buffers ===
// Declare all required static memory buffers (example sizes for MNIST)
float input_tensor[1][1][28][28];
float conv1_out[1][32][26][26];
float relu1_out[1][32][26][26];
float fc1_out[1][10];

// === Tensor Name to Data Pointer Mapping ===
typedef struct {
    const char *name;
    float *data;
} TensorMap;

TensorMap tensor_map[] = {
    {"input_0", (float*)input_tensor},
    {"conv_out", (float*)conv1_out},
    {"relu_out", (float*)relu1_out},
    {"fc_out", (float*)fc1_out},
    {NULL, NULL}
};

float* get_tensor_data(const char *name) {
    for (int i = 0; tensor_map[i].name != NULL; ++i) {
        if (strcmp(tensor_map[i].name, name) == 0)
            return tensor_map[i].data;
    }
    return NULL;
}

// === Operator Stubs ===
void conv2d(float *input, float *weight, float *output) {
    // Implement or call your conv2d logic here
}

void relu(float *input, float *output, int size) {
    for (int i = 0; i < size; i++)
        output[i] = input[i] > 0 ? input[i] : 0;
}

void matmul(float *a, float *b, float *out) {
    // Implement or call your matmul logic here
}

void softmax(float *logits, float *output, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        output[i] = expf(logits[i]);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i)
        output[i] /= sum;
}

// === Main Inference Loop ===
int main() {
    ONNX__ModelProto *model;
    model = onnx__model_proto__unpack(NULL, model_blob_len, model_blob);
    if (!model) {
        fprintf(stderr, "Failed to unpack model.\n");
        return 1;
    }

    ONNX__GraphProto *graph = model->graph;

    for (int i = 0; i < graph->n_node; ++i) {
        ONNX__NodeProto *node = graph->node[i];

        const char *op_type = node->op_type;
        float *input0 = get_tensor_data(node->input[0]);
        float *input1 = node->n_input > 1 ? get_tensor_data(node->input[1]) : NULL;
        float *output = get_tensor_data(node->output[0]);

        if (strcmp(op_type, "Conv") == 0) {
            conv2d(input0, input1, output);
        } else if (strcmp(op_type, "Relu") == 0) {
            relu(input0, output, 32 * 26 * 26);  // Adjust size accordingly
        } else if (strcmp(op_type, "MatMul") == 0) {
            matmul(input0, input1, output);
        } else if (strcmp(op_type, "Softmax") == 0) {
            softmax(input0, output, 10);
        } else {
            printf("Unsupported operator: %s\n", op_type);
        }
    }

    // Output prediction (argmax of final output)
    float *out = get_tensor_data("fc_out");
    int max_i = 0;
    for (int i = 1; i < 10; ++i) {
        if (out[i] > out[max_i]) max_i = i;
    }
    printf("Predicted digit: %d\n", max_i);

    onnx__model_proto__free_unpacked(model, NULL);
    return 0;
}
