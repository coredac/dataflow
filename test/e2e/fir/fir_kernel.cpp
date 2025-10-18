#define NTAPS 32

void kernel(float input[], float output[], float coefficient[])
/*   input :           input sample array */
/*   output:           output sample array */
/*   coefficient:      coefficient array */
{
  int i;
  float sum = 0.0;
//   #pragma clang loop unroll_count(4) vectorize_width(4)
  for (i = 0; i < NTAPS; ++i) {
    sum += input[i] * coefficient[i];
  }
  output[0] = sum;
}
