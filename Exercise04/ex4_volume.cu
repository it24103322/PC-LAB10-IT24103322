#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define NX 500                  // x: columns
#define NY 500                  // y: rows
#define NZ 500                  // z: planes (depth)

#define CHECK(call)                                                   \
    do {                                                              \
        const cudaError_t error = (call);                             \
        if (error != cudaSuccess) {                                   \
            printf("Error: %s:%d, ", __FILE__, __LINE__);             \
            printf("code:%d, reason: %s\n", error,                    \
                   cudaGetErrorString(error));                        \
            exit(1);                                                  \
        }                                                             \
    } while (0)

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

// Element-wise product: C[z][y][x] = A[z][y][x] * B[z][y][x]
__global__ void mulElem3D(int *a, int *b, int *c, int nx, int ny, int nz) {
    // x, y and z coordinates of this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // only compute if x, y AND z are all inside the volume
    if (x < nx && y < ny && z < nz) {
        size_t idx = ((size_t)z * ny + y) * nx + x;     // flat index
        c[idx] = a[idx] * b[idx];
    }
}

void random_ints(int *x, size_t size) {
    for (size_t i = 0; i < size; i++)
        x[i] = rand() % 100;
}

int main(void) {
    size_t count = (size_t)NX * NY * NZ;        // 125,000,000 elements
    size_t size  = count * sizeof(int);         // 500 MB per array
    int *a, *b, *c;                             // host copies (flat arrays)
    int *d_a, *d_b, *d_c;                       // device copies
    double dStart, dElaps;

    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    a = (int *)malloc(size); random_ints(a, count);
    b = (int *)malloc(size); random_ints(b, count);
    c = (int *)malloc(size);

    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // 8 x 8 x 8 = 512 threads per block
    dim3 threadsPerBlock(8, 8, 8);
    // blocks needed in x, y and z, rounded up
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);
    printf("Grid: %d x %d x %d blocks\n", numBlocks.x, numBlocks.y, numBlocks.z);
    printf("Total threads launched = %lld\n",
           (long long)numBlocks.x * numBlocks.y * numBlocks.z * 512);

    dStart = cpuSecond();
    mulElem3D<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, NX, NY, NZ);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    dElaps = cpuSecond() - dStart;

    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // last 1000 elements of C, in memory order
    for (size_t i = count - 1000; i < count; i++) {
        int x = i % NX;
        int y = (i / NX) % NY;
        int z = i / ((size_t)NX * NY);
        printf("C[%d][%d][%d] = %d x %d = %d\n", z, y, x, a[i], b[i], c[i]);
    }

    // check all elements on the CPU
    long long errors = 0;
    for (size_t i = 0; i < count; i++)
        if (c[i] != a[i] * b[i]) errors++;
    printf("Verification: %lld mismatches out of %zu\n", errors, count);
    printf("Time taken - %.7f\n", dElaps);

    free(a); free(b); free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return 0;
}
