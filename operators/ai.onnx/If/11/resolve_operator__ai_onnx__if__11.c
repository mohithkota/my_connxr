//this file was generated by ../../../../../../../../connx/connx_ajit/scripts/onnx_generator/OperatorTypeResolver.py
#include "operator__ai_onnx__if__11.h"
#include "operators/operator_stub.h"
#include <inttypes.h>
#include <stdio.h>

operator_executer
resolve_operator__ai_onnx__if__11(
    node_context *ctx
){
    operator_executer executer = NULL;
    {
    uint32_t B = 0;
if (ctx->inputs[0]) {
    B = ctx->inputs[0]->data_type;
}
    switch ( B ) {
    case 0: //constrained tensor is not set (maybe optional?), just take next case
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL: { executer = (operator_executer) &execute_operator__ai_onnx__if__11__B_tensor_bool; break; }
    default: {
        fprintf(stderr, "no matching type for operator__ai_onnx__if__11 and constraint 'B' with type '%s' found!\n",operator_info_tensorType2str(B));
        break;
    }
}
}
    if (!executer) {
        executer = &operator_stub;
    }
    return executer;
}