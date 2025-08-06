void non_perfect_extra_computation(int input[128][128], int output[128][128]) {
  for (int i = 0; i < 128; i++) {
    int row_sum = 0;
    int row_max = -1000;
    int row_min = 1000;
    int threshold = i * 2;
    int scale = (i % 2 == 0) ? 2 : 3;

    for (int j = 0; j < 128; j++) {
      output[i][j] = input[i][j] * scale;
      row_sum += input[i][j];

      if (input[i][j] > row_max) {
        row_max = input[i][j];
      }
      if (input[i][j] < row_min) {
        row_min = input[i][j];
      }
    }

    int average = row_sum / 128;
    int range = row_max - row_min;
    int normalized = (range > 0) ? (average * 100 / range) : average;

    output[i][0] = average;
    output[i][1] = row_max;
    output[i][2] = row_min;
    output[i][3] = normalized;
    output[i][4] = range;
  }
}