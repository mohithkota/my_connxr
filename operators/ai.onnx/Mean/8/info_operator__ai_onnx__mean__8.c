//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx__mean__8.h"

/* attributes */
static
operator_info_attribute
attributes[] = {

};

/* input tensors */
static
uint32_t
input_tensor_type_data_0[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
inputs[] = {
{
    .name        = "data_0",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = input_tensor_type_data_0
}
};

/* output tensors */
static
uint32_t
output_tensor_type_mean[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16
};

static
operator_info_tensor
outputs[] = {
{
    .name        = "mean",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 3,
    .types       = output_tensor_type_mean
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
info_operator__ai_onnx__mean__8 = {
    .name         = "Mean",
    .range_input  = { 1, 2147483647 },
    .range_output = { 1, 1 },
    .n_attribute  = 0,
    .attribute    = attributes,
    .n_input      = 1,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 1,
    .constraint   = constraints
};