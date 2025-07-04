//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorInfo.py
#include "operators/operator_info.h"
#include "operator__ai_onnx__gridsample__22.h"

/* attributes */
static
operator_info_attribute
attributes[] = {
{
    .name     = "align_corners",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT
},
{
    .name     = "mode",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING
},
{
    .name     = "padding_mode",
    .optional = true,
    .type     = ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING
}
};

/* input tensors */
static
uint32_t
input_tensor_type_X[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16,
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
input_tensor_type_grid[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16,
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
    .constraint  = "T1",
    .n_types     = 16,
    .types       = input_tensor_type_X
},
{
    .name        = "grid",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T2",
    .n_types     = 4,
    .types       = input_tensor_type_grid
}
};

/* output tensors */
static
uint32_t
output_tensor_type_Y[] = {
ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16,
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
    .name        = "Y",
    .optional    = false,
    .variadic    = false,
    .homogeneous = true,
    .constraint  = "T1",
    .n_types     = 16,
    .types       = output_tensor_type_Y
}
};

/* constraints */
static
operator_info_constraint
constraints[] = {
{ "T1" },
{ "T2" }
};

/* operator info */
operator_info
info_operator__ai_onnx__gridsample__22 = {
    .name         = "GridSample",
    .range_input  = { 2, 2 },
    .range_output = { 1, 1 },
    .n_attribute  = 3,
    .attribute    = attributes,
    .n_input      = 2,
    .input        = inputs,
    .n_output     = 1,
    .output       = outputs,
    .n_constraint = 2,
    .constraint   = constraints
};