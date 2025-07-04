//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx__conv__1.h"

/* attributes */
static
operator_info_attribute
attributes[] = {
{
    .name     = "auto_pad",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING
},
{
    .name     = "dilations",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
},
{
    .name     = "group",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT
},
{
    .name     = "kernel_shape",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
},
{
    .name     = "pads",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
},
{
    .name     = "strides",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
}
};

/* input tensors */
static
uint32_t
input_tensor_type_X[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
uint32_t
input_tensor_type_W[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
uint32_t
input_tensor_type_B[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
inputs[] = {
{
    .name        = "X",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = input_tensor_type_X
},
{
    .name        = "W",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = input_tensor_type_W
},
{
    .name        = "B",
    .optional    = true,
    .variadic    = true,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = input_tensor_type_B
}
};

/* output tensors */
static
uint32_t
output_tensor_type_Y[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
outputs[] = {
{
    .name        = "Y",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = output_tensor_type_Y
}
};

/* constraints */
static
operator_info_constraint
constraints[] = {
{ "T" }
};

/* operator info */
operator_info
info_operator__ai_onnx__conv__1 = {
    .name         = "Conv",
    .range_input  = { 2, 3 },
    .range_output = { 1, 1 },
    .n_attribute  = 6,
    .attribute    = attributes,
    .n_input      = 3,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 1,
    .constraint   = constraints
};