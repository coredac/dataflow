using namespace std;

#define M 4
#define N 8
#define K 5

// Example 1: Matrix processing + vectorization + RAW dependency
int irregularLoopExample1() {
  //   vector<vector<int>> A(M, vector<int>(N, 0));
  int A[M][N];
  int B[M][N];
  int temp[N];

  for (int i = 0; i < M; i++) {
    // First independent loop: matrix initialization (Independent Loop 1)
    for (int j = 0; j < N; j++) {
      A[i][j] = i * N + j;
      temp[j] = 0; // Initialize temp
    }

    // Non-nested code segment
    int sum = 0;
    for (int k = 0; k < K; k++) {
      sum += k;
    }

    // Second independent loop: using the results of the first loop (Independent
    // Loop 2 - RAW Dependency) RAW: depends on the writes to temp[j] above
    for (int j = 0; j < N; j++) {
      B[i][j] = A[i][j] + temp[j] + sum; // Read temp[j] (RAW dependency)
      B[i][j] *= 2;
    }
  }
  return B[M - 1][N - 1];
}