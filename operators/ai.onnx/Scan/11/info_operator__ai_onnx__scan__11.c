//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx__scan__11.h"

/* attributes */
static
operator_info_attribute
attributes[] = {
{
    .name     = "body",
    .optional = false,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH
},
{
    .name     = "num_scan_inputs",
    .optional = false,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT
},
{
    .name     = "scan_input_axes",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
},
{
    .name     = "scan_input_directions",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
},
{
    .name     = "scan_output_axes",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
},
{
    .name     = "scan_output_directions",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS
}
};

/* input tensors */
static
uint32_t
input_tensor_type_initial_state_and_scan_inputs[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BOOL,
ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128,
ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64,
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16,
ONNX__TENSOR_PROTO__DATA_TYPE__INT16,
ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
ONNX__TENSOR_PROTO__DATA_TYPE__INT64,
ONNX__TENSOR_PROTO__DATA_TYPE__INT8,
ONNX__TENSOR_PROTO__DATA_TYPE__STRING,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT16,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT32,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT64,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT8
};

static
operator_info_tensor
inputs[] = {
{
    .name        = "initial_state_and_scan_inputs",
    .optional    = false,
    .variadic    = false,
    .homogeneous = false,
    .constraint  = "V",
    .n_types     = 15,
    .types       = input_tensor_type_initial_state_and_scan_inputs
}
};

/* output tensors */
static
uint32_t
output_tensor_type_final_state_and_scan_outputs[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BOOL,
ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128,
ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64,
ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16,
ONNX__TENSOR_PROTO__DATA_TYPE__INT16,
ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
ONNX__TENSOR_PROTO__DATA_TYPE__INT64,
ONNX__TENSOR_PROTO__DATA_TYPE__INT8,
ONNX__TENSOR_PROTO__DATA_TYPE__STRING,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT16,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT32,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT64,
ONNX__TENSOR_PROTO__DATA_TYPE__UINT8
};

static
operator_info_tensor
outputs[] = {
{
    .name        = "final_state_and_scan_outputs",
    .optional    = false,
    .variadic    = false,
    .homogeneous = false,
    .constraint  = "V",
    .n_types     = 15,
    .types       = output_tensor_type_final_state_and_scan_outputs
}
};

/* constraints */
static
operator_info_constraint
constraints[] = {
{ "V" }
};

/* operator info */
operator_info
info_operator__ai_onnx__scan__11 = {
    .name         = "Scan",
    .range_input  = { 1, 2147483647 },
    .range_output = { 1, 2147483647 },
    .n_attribute  = 6,
    .attribute    = attributes,
    .n_input      = 1,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 1,
    .constraint   = constraints
};