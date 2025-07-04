//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTypeResolver.py
#include "operator__ai_onnx__qlinearconv__10.h"
#include "operators/operator_stub.h"
#include <inttypes.h>
#include <stdio.h>

operator_executer
resolve_operator__ai_onnx__qlinearconv__10(
    node_context *ctx
){
    operator_executer executer = NULL;
    {
    uint32_t T1 = 0;
if (ctx->inputs[0]) {
    T1 = ctx->inputs[0]->data_type;
}
uint32_t T2 = 0;
if (ctx->inputs[3]) {
    T2 = ctx->inputs[3]->data_type;
}
uint32_t T3 = 0;
if (ctx->inputs[7]) {
    T3 = ctx->inputs[7]->data_type;
}
uint32_t T4 = 0;
if (ctx->inputs[8]) {
    T4 = ctx->inputs[8]->data_type;
}
    switch ( T1 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: { switch ( T2 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: { switch ( T3 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_int8__T2_tensor_int8__T3_tensor_int8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_int8__T2_tensor_int8__T3_tensor_uint8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T3' with type '%s' found!\n",operator_info_tensorType2str(T3));
        break;
    }
} break; }
case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: { switch ( T3 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_int8__T2_tensor_uint8__T3_tensor_int8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_int8__T2_tensor_uint8__T3_tensor_uint8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T3' with type '%s' found!\n",operator_info_tensorType2str(T3));
        break;
    }
} break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T2' with type '%s' found!\n",operator_info_tensorType2str(T2));
        break;
    }
} break; }
case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: { switch ( T2 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: { switch ( T3 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_uint8__T2_tensor_int8__T3_tensor_int8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_uint8__T2_tensor_int8__T3_tensor_uint8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T3' with type '%s' found!\n",operator_info_tensorType2str(T3));
        break;
    }
} break; }
case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: { switch ( T3 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_uint8__T2_tensor_uint8__T3_tensor_int8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: { switch ( T4 ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer) &execute_operator__ai_onnx__qlinearconv__10__T1_tensor_uint8__T2_tensor_uint8__T3_tensor_uint8__T4_tensor_int32; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T4' with type '%s' found!\n",operator_info_tensorType2str(T4));
        break;
    }
} break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T3' with type '%s' found!\n",operator_info_tensorType2str(T3));
        break;
    }
} break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T2' with type '%s' found!\n",operator_info_tensorType2str(T2));
        break;
    }
} break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__qlinearconv__10 and constraint 'T1' with type '%s' found!\n",operator_info_tensorType2str(T1));
        break;
    }
}
}
    if (!executer) {
        executer = &operator_stub;
    }
    return executer;
}