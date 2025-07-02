#include <stdio.h>
#include <string.h>
#include <math.h>

// ----------- Reshape -----------
void Reshape(const float* input, float* output, int total_size) {
    for (int i = 0; i < total_size; ++i)
        output[i] = input[i];
}

// ----------- Conv2D -----------
void Conv2D(
    const float* input, int in_h, int in_w, int in_c,
    const float* weights, int out_c, int kernel_h, int kernel_w,
    const float* bias,
    float* output, int out_h, int out_w
) {
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = bias ? bias[oc] : 0;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            int input_idx = ((ih * in_w + iw) * in_c + ic);
                            int weight_idx = (((oc * in_c + ic) * kernel_h + kh) * kernel_w + kw);
                            sum += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
                int out_idx = ((oh * out_w + ow) * out_c + oc);
                output[out_idx] = sum;
            }
        }
    }
}

// ----------- ReLU -----------
void Relu(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i)
        output[i] = input[i] > 0 ? input[i] : 0;
}

// ----------- MaxPool2D -----------
void MaxPool2D(
    const float* input, float* output,
    int in_h, int in_w, int channels,
    int pool_h, int pool_w, int stride
) {
    int out_h = (in_h - pool_h) / stride + 1;
    int out_w = (in_w - pool_w) / stride + 1;

    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float max_val = -1e10;
                for (int ph = 0; ph < pool_h; ++ph) {
                    for (int pw = 0; pw < pool_w; ++pw) {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;
                        int input_idx = ((ih * in_w + iw) * channels + c);
                        if (input[input_idx] > max_val)
                            max_val = input[input_idx];
                    }
                }
                int output_idx = ((oh * out_w + ow) * channels + c);
                output[output_idx] = max_val;
            }
        }
    }
}

// ----------- Transpose2D -----------
void Transpose2D(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            output[c * rows + r] = input[r * cols + c];
}

// ----------- MatMul -----------
void MatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ----------- Add -----------
void Add(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; ++i)
        C[i] = A[i] + B[i];
}

// ----------- Softmax -----------
void Softmax(const float* input, float* output, int size) {
    float max = input[0];
    for (int i = 1; i < size; ++i)
        if (input[i] > max) max = input[i];

    float sum = 0;
    for (int i = 0; i < size; ++i) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < size; ++i)
        output[i] /= sum;
}

