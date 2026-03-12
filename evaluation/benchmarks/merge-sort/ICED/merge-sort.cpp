#include <cstdint>
#define SIZE 1024

void merge(int64_t a[SIZE], int64_t k) {
  int64_t start = 0, m = SIZE, stop = SIZE;
  int64_t temp[SIZE];
  int64_t i, j;

  i = start;
  j = stop;

  // for (k = start; k <= stop; k++) {
  int64_t tmp_j = temp[j];
  int64_t tmp_i = temp[i];
  if (tmp_j < tmp_i) {
    a[k] = tmp_j;
    j--;
  } else {
    a[k] = tmp_i;
    i++;
  }
  // }
}