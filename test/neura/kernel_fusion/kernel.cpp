// Test cases for FuseKernelPass
//
// Build workflow using Polygeist:
//   1. cgeist kernel_fusion.cpp -S -O2       -> SCF loops (kernel_fusion_scf.mlir)
//   2. polygeist-opt --raise-scf-to-affine   -> Affine loops (kernel_fusion_affine.mlir)
//   3. mlir-neura-opt --wrap-loop-in-kernel  -> neura.kernel ops (kernel_fusion_wrapped.mlir)
//   4. mlir-neura-opt --fuse-kernel          -> Fused kernels (kernel_fusion_fused.mlir)

#define N 64

float A[N], B[N], C[N], D[N], E[N], F[N], G[N], H[N], X[N], Y[N];

// Producer-Consumer Fusion: kernel0 -> kernel1
// kernel0: C[i] = A[i] + B[i]
// kernel1: D[i] = C[i] * 2.0
void test_producer_consumer_fusion(float A[], float B[], float C[], float D[]) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < N; i++) {
        D[i] = C[i] * 2.0f;
    }
}

// Multiple Consumers: kernel0 -> kernel1, kernel0 -> kernel2
// kernel0: C[i] = A[i] + B[i]
// kernel1: D[i] = C[i] * 2.0
// kernel2: E[i] = C[i] + 1.0
void test_multiple_consumers(float A[], float B[], float C[], float D[], float E[]) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < N; i++) {
        D[i] = C[i] * 2.0f;
    }

    for (int i = 0; i < N; i++) {
        E[i] = C[i] + 1.0f;
    }
}

// Sibling Fusion: kernel0 || kernel1 (share input A)
// kernel0: E[i] = A[i] * 3.0
// kernel1: F[i] = A[i] + 1.0
void test_sibling_fusion(float A[], float E[], float F[]) {
    for (int i = 0; i < N; i++) {
        E[i] = A[i] * 3.0f;
    }

    for (int i = 0; i < N; i++) {
        F[i] = A[i] + 1.0f;
    }
}

// No Shared Input: kernel0, kernel1 (no fusion - different inputs)
// kernel0: G[i] = X[i] * 2.0
// kernel1: H[i] = Y[i] + 3.0
void test_no_shared_input(float X[], float Y[], float G[], float H[]) {
    for (int i = 0; i < N; i++) {
        G[i] = X[i] * 2.0f;
    }

    for (int i = 0; i < N; i++) {
        H[i] = Y[i] + 3.0f;
    }
}

// Chain Fusion: kernel0 -> kernel1 -> kernel2
// kernel0: C[i] = A[i] + B[i]
// kernel1: D[i] = C[i] * 2.0
// kernel2: E[i] = D[i] + 1.0
void test_chain_fusion(float A[], float B[], float C[], float D[], float E[]) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < N; i++) {
        D[i] = C[i] * 2.0f;
    }

    for (int i = 0; i < N; i++) {
        E[i] = D[i] + 1.0f;
    }
}

// Complex Sibling: (kernel0 || kernel1 || kernel2), kernel3
// kernel0: C[i] = A[i] * 2.0
// kernel1: D[i] = A[i] + 1.0  } siblings (share A)
// kernel2: E[i] = A[i] - 1.0
// kernel3: F[i] = B[i] * 3.0  (independent)
void test_complex_sibling(float A[], float B[], float C[], float D[], float E[], float F[]) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * 2.0f;
    }

    for (int i = 0; i < N; i++) {
        D[i] = A[i] + 1.0f;
    }

    for (int i = 0; i < N; i++) {
        E[i] = A[i] - 1.0f;
    }

    for (int i = 0; i < N; i++) {
        F[i] = B[i] * 3.0f;
    }
}

// Mixed Patterns: (kernel0 -> kernel3) || (kernel1 || kernel2)
// kernel0: C[i] = A[i] + B[i] ─┐
// kernel1: D[i] = A[i] * 2.0   ├─ siblings (share A)
// kernel2: E[i] = A[i] + 3.0  ─┘
// kernel3: F[i] = C[i] * 2.0    (consumer of kernel0)
void test_mixed_patterns(float A[], float B[], float C[], float D[], float E[], float F[]) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < N; i++) {
        D[i] = A[i] * 2.0f;
    }

    for (int i = 0; i < N; i++) {
        E[i] = A[i] + 3.0f;
    }

    for (int i = 0; i < N; i++) {
        F[i] = C[i] * 2.0f;
    }
}

int main() {
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
        X[i] = (float)(i + 1);
        Y[i] = (float)(i - 1);
    }

    test_producer_consumer_fusion(A, B, C, D);
    test_sibling_fusion(A, E, F);
    test_no_shared_input(X, Y, G, H);
    test_chain_fusion(A, B, C, D, E);
    test_complex_sibling(A, B, C, D, E, F);
    test_mixed_patterns(A, B, C, D, E, F);

    return 0;
}
