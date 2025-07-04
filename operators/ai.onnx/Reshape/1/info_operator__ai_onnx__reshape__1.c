//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx__reshape__1.h"

/* attributes */
static
operator_info_attribute
attributes[] = {
{
    .name     = "consumed_inputs",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
},
{
    .name     = "shape",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
}
};

/* input tensors */
static
uint32_t
input_tensor_type_data[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
inputs[] = {
{
    .name        = "data",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = input_tensor_type_data
}
};

/* output tensors */
static
uint32_t
output_tensor_type_reshaped[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
outputs[] = {
{
    .name        = "reshaped",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = output_tensor_type_reshaped
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
info_operator__ai_onnx__reshape__1 = {
    .name         = "Reshape",
    .range_input  = { 1, 1 },
    .range_output = { 1, 1 },
    .n_attribute  = 2,
    .attribute    = attributes,
    .n_input      = 1,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 1,
    .constraint   = constraints
};