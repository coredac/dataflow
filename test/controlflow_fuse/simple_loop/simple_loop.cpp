void simple_loop(int data[128], int output[128]) {
  for (int i = 0; i < 128; ++i) {
    output[i] = data[i] * 2 + 1;
  }
}