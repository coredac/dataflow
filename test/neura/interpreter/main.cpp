#include <cstdio>

extern "C" float test();

int main() {
  float result = test();
  std::printf("Golden output: %f\n", result);
  return 0;
}

