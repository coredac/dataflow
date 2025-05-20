// RUN: mlir-neura-opt %s | FileCheck %s

#include <stdio.h>

#define NTAPS 32

float input[NTAPS] = {
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
};
float output[NTAPS];
float coefficients[NTAPS] = {0.25, 1.50, 3.75, -2.25, 0.50, 0.75, -3.00, 1.25,
0.25, 1.50, 3.75, -2.25, 0.50, 0.75, -3.00, 1.25,
0.25, 1.50, 3.75, -2.25, 0.50, 0.75, -3.00, 1.25,
0.25, 1.50, 3.75, -2.25, 0.50, 0.75, -3.00, 1.25};

void kernel(float input[], float output[], float coefficient[]);

int main()
{

//  input_dsp (input, NTAPS, 0);

  kernel(input, output, coefficients);

//  output_dsp (input, NTAPS, 0);
//  output_dsp (coefficients, NTAPS, 0);
//  output_dsp (output, NTAPS, 0);
  printf("output: %f\n", output[0]);
  return 0;
}

/*   input :           input sample array */
/*   output:           output sample array */
/*   coefficient:      coefficient array */
void kernel(float input[], float output[], float coefficient[]) {
  int i;
  int j = 0;

   for (i = 0; i < NTAPS; ++i) {
     float tmp = input[i] * coefficient[i];
     output[j] += tmp;
   }
}
