#define DATA_LEN 20
#define BUCKET_LEN 5
#define MIN 1.0
#define MAX 19.0

void kernel(float input[], int histogram[]) {
  int i;
  float dmin = (float)MIN;
  float delt = (float)(MAX - dmin);

//   #pragma clang loop vectorize(enable) vectorize_width(4) unroll_count(4)
  for (i = 0; i < DATA_LEN; i++) {
    float r = BUCKET_LEN * (input[i] - dmin) / delt;
    int b = (int)(r);
    histogram[b]++;
  }
}
