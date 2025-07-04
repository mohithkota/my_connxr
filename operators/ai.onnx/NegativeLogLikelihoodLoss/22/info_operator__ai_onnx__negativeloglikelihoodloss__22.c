//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx__negativeloglikelihoodloss__22.h"

/* attributes */
static
operator_info_attribute
attributes[] = {
{
    .name     = "ignore_index",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT
},
{
    .name     = "reduction",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING
}
};

/* input tensors */
static
uint32_t
input_tensor_type_input[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16,
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
uint32_t
input_tensor_type_target[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
ONNX__TENSOR_PROTO__DATA_TYPE__INT64
};

static
uint32_t
input_tensor_type_weight[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16,
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
inputs[] = {
{
    .name        = "input",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 4,
    .types       = input_tensor_type_input
},
{
    .name        = "target",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "Tind",
    .n_types     = 2,
    .types       = input_tensor_type_target
},
{
    .name        = "weight",
    .optional    = true,
    .variadic    = true,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 4,
    .types       = input_tensor_type_weight
}
};

/* output tensors */
static
uint32_t
output_tensor_type_loss[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16,
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
outputs[] = {
{
    .name        = "loss",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 4,
    .types       = output_tensor_type_loss
}
};

/* constraints */
static
operator_info_constraint
constraints[] = {
{ "T" },
{ "Tind" }
};

/* operator info */
operator_info
info_operator__ai_onnx__negativeloglikelihoodloss__22 = {
    .name         = "NegativeLogLikelihoodLoss",
    .range_input  = { 2, 3 },
    .range_output = { 1, 1 },
    .n_attribute  = 2,
    .attribute    = attributes,
    .n_input      = 3,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 2,
    .constraint   = constraints
};