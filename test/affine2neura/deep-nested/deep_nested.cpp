int input_data[3][3][3];
int output_data[3][3][3];
float weights[3];

int deep_nested() {
  // 10 nested loops
  for (int i0 = 0; i0 < 3; i0++) {
    for (int i1 = 0; i1 < 3; i1++) {
      for (int i2 = 0; i2 < 3; i2++) {
        for (int i3 = 0; i3 < 3; i3++) {
          for (int i4 = 0; i4 < 3; i4++) {
            for (int i5 = 0; i5 < 3; i5++) {
              for (int i6 = 0; i6 < 3; i6++) {
                for (int i7 = 0; i7 < 3; i7++) {
                  for (int i8 = 0; i8 < 3; i8++) {
                    for (int i9 = 0; i9 < 3; i9++) {
                      // Assuming some operation on input_data
                      output_data[i0][i1][i2] +=
                          input_data[i0][i1][i2];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
