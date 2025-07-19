int perfect_nested_reduction_2d(int matrix[128][128]) {
  int sum = 0;
  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 128; j++) {
      sum += matrix[i][j];
    }
  }
  return sum;
}