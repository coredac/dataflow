float A[100];
float C[100];

int main() {
  const int size = 100;
  for (int i = 0; i < size; ++i) {
    float loaded_value = A[i];      // Instruction 1: Load value from A
    float multiplied_value = loaded_value * 10.0f; // Instruction 2: Multiply the value
    C[i] = multiplied_value;      // Instruction 3: Store result into C
  }
  return 0;
}
