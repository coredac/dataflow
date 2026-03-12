#include <cstdint>

#define C_IN 64
#define W_IN 224
#define C_OUT 128
#define K 3
#define W_OUT (W_IN - K + 1)

void conv1d(int64_t *output, int64_t *input, int64_t *kernel) {
  // for (int64_t output_pos = 0; output_pos < C_OUT * W_OUT; ++output_pos) {
  int64_t output_pos = 0;
  int64_t c_out = output_pos / W_OUT;
  int64_t w_out = output_pos % W_OUT;
  int64_t acc = 0;

  for (int64_t kernel_pos = 0; kernel_pos < C_IN * K; ++kernel_pos) {
    int64_t c_in = kernel_pos / K;
    int64_t kw = kernel_pos % K;

    int64_t w_in_idx = w_out + kw;
    int64_t input_idx = c_in * W_IN + w_in_idx;
    int64_t kernel_idx = c_out * (C_IN * K) + c_in * K + kw;

    acc += input[input_idx] * kernel[kernel_idx];
  }

  output[output_pos] = acc;
  // }
}