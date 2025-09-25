#include <stdio.h>

#define N 32

int input[N] = {1,   -1,  2,   -3,  4,   -5,  6,   -7,  8,   -9, 10,
                -11, 12,  -13, 14,  -15, 16,  -17, 18,  -19, 20, -21,
                22,  -23, 24,  -25, 26,  -27, 28,  -29, 30,  -31};

int output[N];

void kernel(int input[], int output[]);

int main() {
  // init output
  for (int i = 0; i < N; i++) {
    output[i] = 0;
  }

  kernel(input, output);

  // print outputs
  for (int i = 0; i < N; i++) {
    printf("output[%d] = %d\n", i, output[i]);
  }

  return 0;
}

void kernel(int input[], int output[]) {
  for (int i = 0; i < N; ++i) {
    if (input[i] > 0) {
      output[i] += input[i];
    } else {
      output[i] += 0;
    }
  }
}