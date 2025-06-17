float input[1][16][64];
float output[1][16];

int node11() {
  for (int arg2 = 0; arg2 < 1; arg2++) {
    for (int arg3 = 0; arg3 < 16; arg3++) {
      for (int arg4 = 0; arg4 < 64; arg4+=1) 
        output[arg2][arg3] += input[arg2][arg3][arg4];
    }
  }
  return 0;
}