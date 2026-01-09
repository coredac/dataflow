// Pure nested loop structure - no inter-loop computations
int pureNestedLoopExample(int d1[4][8][6], int d2[4][8][5], int d3[4][8][5],
                          int d4[4][7], int d5[4][9], int m1[6], int m2[5],
                          int m3[7], int m4[9], int *result) {
  for (int i = 0; i < 4; i++) {     // Loop A
    for (int j = 0; j < 8; j++) {   // Loop B
      for (int k = 0; k < 6; k++) { // Loop C
        m1[k] = d1[i][j][k];
      }
      for (int k = 0; k < 5; k++) { // Loop D
        m2[k] = d2[i][j][k] + d3[i][j][k];
      }
      for (int k = 0; k < 6; k++) { // Loop E
        *result += m1[k] + m2[k];
      }
    }
    for (int j = 0; j < 7; j++) { // Loop F
      m3[j] = d4[i][j];
    }
    for (int j = 0; j < 9; j++) { // Loop G
      m4[j] = d5[i][j] + m3[j];
    }
  }
  return *result;
}