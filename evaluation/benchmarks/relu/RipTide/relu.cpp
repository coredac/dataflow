#include <cstdint>
#define N 512

void relu(int64_t input[], int64_t output[], int64_t i) {
  // for (int64_t i = 0; i < N; ++i) {
  if (input[i] > 0) {
    output[i] += input[i];
  } else {
    output[i] += 0;
  }
  // }
}