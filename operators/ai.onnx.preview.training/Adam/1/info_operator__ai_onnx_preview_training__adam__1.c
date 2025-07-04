//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx_preview_training__adam__1.h"

/* attributes */
static
operator_info_attribute
attributes[] = {
{
    .name     = "alpha",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT
},
{
    .name     = "beta",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT
},
{
    .name     = "epsilon",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT
},
{
    .name     = "norm_coefficient",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT
},
{
    .name     = "norm_coefficient_post",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT
}
};

/* input tensors */
static
uint32_t
input_tensor_type_R[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT
};

static
uint32_t
input_tensor_type_T[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__INT64
};

static
uint32_t
input_tensor_type_inputs[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT
};

static
operator_info_tensor
inputs[] = {
{
    .name        = "R",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T1",
    .n_types     = 2,
    .types       = input_tensor_type_R
},
{
    .name        = "T",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T2",
    .n_types     = 1,
    .types       = input_tensor_type_T
},
{
    .name        = "inputs",
    .optional    = false,
    .variadic    = false,
    .homogeneous = false,
    .constraint  = "T3",
    .n_types     = 2,
    .types       = input_tensor_type_inputs
}
};

/* output tensors */
static
uint32_t
output_tensor_type_outputs[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT
};

static
operator_info_tensor
outputs[] = {
{
    .name        = "outputs",
    .optional    = false,
    .variadic    = false,
    .homogeneous = false,
    .constraint  = "T3",
    .n_types     = 2,
    .types       = output_tensor_type_outputs
}
};

/* constraints */
static
operator_info_constraint
constraints[] = {
{ "T1" },
{ "T2" },
{ "T3" }
};

/* operator info */
operator_info
info_operator__ai_onnx_preview_training__adam__1 = {
    .name         = "Adam",
    .range_input  = { 3, 2147483647 },
    .range_output = { 1, 2147483647 },
    .n_attribute  = 5,
    .attribute    = attributes,
    .n_input      = 3,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 3,
    .constraint   = constraints
};