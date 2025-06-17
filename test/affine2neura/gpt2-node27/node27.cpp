float input[1][16][4][16];
float output[1][4][16][16];

int node27() {
  for (int arg2 = 0; arg2 < 1; arg2++) {
    for (int arg3 = 0; arg3 < 16; arg3++) {
      for (int arg4 = 0; arg4 < 4; arg4 += 1) {
        for (int arg5 = 0; arg5 < 16; arg5 += 1) {
          output[arg2][arg3][arg4][arg5] = input[arg2][arg4][arg3][arg5];
        }
      }
    }
  }
}