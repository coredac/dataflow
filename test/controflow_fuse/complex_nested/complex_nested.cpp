// This function is used in image processing.
void complex_nested(int cube[32][32][32], int result[32][32]) {
  for (int i = 0; i < 32; i++) {
    int plane_sum = 0;

    for (int j = 0; j < 32; j++) {
      result[i][j] = 0;
      for (int k = 0; k < 32; k++) {
        result[i][j] += cube[i][j][k];
      }
    }

    int avg_value = 0;
    for (int j = 0; j < 32; j++) {
      plane_sum += result[i][j];
    }
    avg_value = plane_sum / 32;

    for (int j = 0; j < 32; j++) {
      int column_max = -128;
      for (int k = 0; k < 32; k++) {
        if (cube[k][j][i] > column_max) {
          column_max = cube[k][j][i];
        }
      }
      result[i][j] = (result[i][j] * column_max) / 128;
    }

    for (int j = 0; j < 32; j++) {
      if (result[i][j] > avg_value) {
        result[i][j] = avg_value;
      }
    }
  }
}