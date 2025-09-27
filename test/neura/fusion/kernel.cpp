#include <stdio.h>
#include <stdlib.h>

#define NTAPS 1024

int A[NTAPS][NTAPS];
int s[NTAPS];
int q[NTAPS];
int p[NTAPS];
int r[NTAPS];

void kernel(int A[][NTAPS], int s[], int q[], int p[], int r[]) {
    int i, j;

    for (i = 0; i < NTAPS; i++) {
        for (j = 0; j < NTAPS; j++) {
            s[j] = s[j] + r[i] * A[i][j];
            q[i] = q[i] + A[i][j] * p[j];
        }
    }
}

int main() {
    kernel(A, s, q, p, r);
}

