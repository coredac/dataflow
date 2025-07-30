int simpleloop() {
  int start = 0;
  int multiplier = 1;
  int result = start;
  for (int i = 0; i < 128; i++) {
    result = result * multiplier + i;
  }

  return result;
}