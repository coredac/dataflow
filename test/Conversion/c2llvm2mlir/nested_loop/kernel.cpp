// RUN: mlir-neura-opt %s | FileCheck %s

#include <stdio.h>

#define NTAPS 32

int input[NTAPS] = {
1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1
};
int output[NTAPS];
int coefficients[NTAPS] = {25, 150, 375, -225, 50, 75, -300, 125,
25, 150, 375, -225, 50, 75, -300, 125,
25, 150, 375, -225, 50, 75, -300, 125,
25, 150, 375, -225, 50, 75, -300, 125};

void kernel(int input[], int output[], int coefficient[]);

int main()
{

//  input_dsp (input, NTAPS, 0);

  kernel(input, output, coefficients);

//  output_dsp (input, NTAPS, 0);
//  output_dsp (coefficients, NTAPS, 0);
//  output_dsp (output, NTAPS, 0);
  printf("output: %d\n", output[0]);
  return 0;
}

/*   input :           input sample array */
/*   output:           output sample array */
/*   coefficient:      coefficient array */
void kernel(int input[], int output[], int coefficient[]) {
  int i, j;

   for (i = 0; i < NTAPS; ++i) {
        for (j = 0; j < NTAPS; ++j) {
            output[j] += input[i] * coefficient[i];
        }
    }
}
