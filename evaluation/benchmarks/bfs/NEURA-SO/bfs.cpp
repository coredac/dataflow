#define N 256
#include <cstdint>

void bfs(int64_t ofs[N], int64_t nbrs[], int64_t parents[N], int64_t queue[N]) {
  // int64_t source;

  // int64_t head = 0;
  int64_t tail = 1;

  // Mark source as visited
  // parents[source] = source;
  // queue[0] = source;

  // Single-loop BFS implementation
  // for (int64_t current_head = 0; current_head < tail; current_head++) {
  int64_t current_head = 10;
  int64_t v = queue[current_head];

  // Process each neighbor
  int64_t start = ofs[v];
  int64_t end = ofs[v + 1];
  int64_t i = start;

  // for (int64_t i = start; i < end; i++) {
  int64_t nb = nbrs[i];

  // Process only if unvisited
  if (parents[nb] == -1) {
    parents[nb] = v;
    queue[tail] = nb;
    tail++;
  }
  // }
  // }
}