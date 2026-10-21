#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 100000000             // 100 million elements (about 400 MB per array)
#define THREADS_PER_BLOCK 512

__global__ void vecMul(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b[i];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

// Track CPU (wall-clock) time, in seconds
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void) {
    int *a, *b, *c;             // host copies of a, b, c
    int *d_a, *d_b, *d_c;       // device copies of a, b, c
    size_t size = (size_t)N * sizeof(int);
    double wStart, wElaps, kStart, kElaps;

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // whole sequence: copy in -> kernel -> copy out
    wStart = cpuSecond();
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // kernel only
    kStart = cpuSecond();
    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    kElaps = cpuSecond() - kStart;

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    wElaps = cpuSecond() - wStart;

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (c[i] != a[i] * b[i]) errors++;
    printf("Verification: %d mismatches out of %d\n", errors, N);
    printf("Kernel only                 - %.7f\n", kElaps);
    printf("Copy in + kernel + copy out - %.7f\n", wElaps);
    printf("Whole sequence / kernel     = %.1f times\n", wElaps / kElaps);

    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
