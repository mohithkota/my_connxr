#include "onnx.pb-c.h"
#include "tensor_store.h"
#include "input_blob.h"
#include <stdio.h>

void load_input_tensor_from_blob() {
    ProtobufCAllocator allocator = {
        .alloc = static_alloc,
        .free = NULL,
        .allocator_data = NULL
    };

    Onnx__TensorProto* tensor = onnx__tensor_proto__unpack(&allocator, input_tensor_len, input_tensor);
    if (!tensor) {
        printf("❌ Failed to unpack input tensor\n");
        return;
    }

    Tensor* t = create_tensor(tensor->name);
    t->dim_count = tensor->n_dims;
    t->size = 1;

    for (size_t i = 0; i < t->dim_count; i++) {
        t->dims[i] = tensor->dims[i];
        t->size *= t->dims[i];
    }

    if (tensor->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT) {
        for (size_t i = 0; i < t->size; i++) {
            t->data[i] = tensor->float_data[i];
        }
    } else {
        printf("❌ Unsupported data type: %d\n", tensor->data_type);
    }

    printf("✅ Loaded input tensor: %s [%zu values]\n", t->name, t->size);
}

