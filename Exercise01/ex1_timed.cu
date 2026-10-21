#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

__global__ void add(int *a, int *b, int *c) {
    // each block runs only one addition
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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
    int size = N * sizeof(int);
    double dStart, dElaps;

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a, b, c and set up input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // initialize timer
    dStart = cpuSecond();
    // Launch add() kernel on GPU with N blocks
    add<<<N, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();    // wait for the GPU to finish, THEN stop the clock
    dElaps = cpuSecond() - dStart;

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int r = 0; r < 5; r++)
        printf("%d + %d = %d\n", a[r], b[r], c[r]);

    // Check the GPU's answers against the CPU
    int errors = 0;
    for (int r = 0; r < N; r++)
        if (c[r] != a[r] + b[r]) errors++;
    printf("Verification: %d mismatches out of %d\n", errors, N);
    printf("Time taken - %.7f\n", dElaps);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
