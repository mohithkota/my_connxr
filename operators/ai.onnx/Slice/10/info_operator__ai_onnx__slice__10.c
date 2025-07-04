//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx__slice__10.h"

/* attributes */
static
operator_info_attribute
attributes[] = {

};

/* input tensors */
static
uint32_t
input_tensor_type_data[] = {
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
uint32_t
input_tensor_type_starts[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
ONNX__TENSOR_PROTO__DATA_TYPE__INT64
};

static
uint32_t
input_tensor_type_ends[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
ONNX__TENSOR_PROTO__DATA_TYPE__INT64
};

static
uint32_t
input_tensor_type_axes[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
ONNX__TENSOR_PROTO__DATA_TYPE__INT64
};

static
uint32_t
input_tensor_type_steps[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
ONNX__TENSOR_PROTO__DATA_TYPE__INT64
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
    .n_types     = 15,
    .types       = input_tensor_type_data
},
{
    .name        = "starts",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "Tind",
    .n_types     = 2,
    .types       = input_tensor_type_starts
},
{
    .name        = "ends",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "Tind",
    .n_types     = 2,
    .types       = input_tensor_type_ends
},
{
    .name        = "axes",
    .optional    = true,
    .variadic    = true,
    .homogeneous = true,
    .constraint  = "Tind",
    .n_types     = 2,
    .types       = input_tensor_type_axes
},
{
    .name        = "steps",
    .optional    = true,
    .variadic    = true,
    .homogeneous = true,
    .constraint  = "Tind",
    .n_types     = 2,
    .types       = input_tensor_type_steps
}
};

/* output tensors */
static
uint32_t
output_tensor_type_output[] = {
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
    .name        = "output",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T",
    .n_types     = 15,
    .types       = output_tensor_type_output
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
info_operator__ai_onnx__slice__10 = {
    .name         = "Slice",
    .range_input  = { 3, 5 },
    .range_output = { 1, 1 },
    .n_attribute  = 0,
    .attribute    = attributes,
    .n_input      = 5,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 2,
    .constraint   = constraints
};