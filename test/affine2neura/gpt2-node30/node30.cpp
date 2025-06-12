float A[1][4][16][64];
// float B=20.0;
float C[1][4][16][64];

int main() {
  for (int arg2 = 0; arg2 < 1; arg2++) {
    for (int arg3 = 0; arg3 < 4; arg3++) {
      for (int arg4 = 0; arg4 < 16; arg4++) {
        for (int arg5 = 0; arg5 < 64; arg5++) {
          C[arg2][arg3][arg4][arg5] = A[arg2][arg3][arg4][arg5] * 10;
        }
      }
    }
  }
}