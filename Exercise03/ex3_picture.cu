#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define WIDTH  800              // columns
#define HEIGHT 600              // rows

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

// Kernel from the lecture slide "Source Code of a PictureKernel", adapted
// for 8-bit pixels: scale every pixel value by 2.0, but never exceed 255.
__global__ void PictureKernel(unsigned char *d_Pin, unsigned char *d_Pout,
                              int height, int width) {
    // Calculate the row # of the d_Pin and d_Pout element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_Pin and d_Pout element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread computes one element of d_Pout if in range
    if ((Row < height) && (Col < width)) {
        int v = 2 * d_Pin[Row * width + Col];
        d_Pout[Row * width + Col] = (v > 255) ? 255 : v;
    }
}

int main(void) {
    size_t count = (size_t)WIDTH * HEIGHT;      // ONE flat array: 480,000 pixels
    size_t size  = count * sizeof(unsigned char);
    unsigned char *h_in, *h_out;                // host: original and updated picture
    unsigned char *d_in, *d_out;                // device copies
    double dStart, dElaps;

    h_in  = (unsigned char *)malloc(size);
    h_out = (unsigned char *)malloc(size);
    CHECK(cudaMalloc((void **)&d_in,  size));
    CHECK(cudaMalloc((void **)&d_out, size));

    // every pixel gets a random value between 0 and 255
    for (size_t i = 0; i < count; i++)
        h_in[i] = rand() % 256;

    CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // 16 threads per block in BOTH x and y
    dim3 threadsPerBlock(16, 16);
    // blocks needed in x (columns) and in y (rows), rounded up
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    printf("Grid: %d x %d blocks of %d x %d threads\n",
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    printf("Total threads launched = %d\n",
           numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y);

    dStart = cpuSecond();
    PictureKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, HEIGHT, WIDTH);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    dElaps = cpuSecond() - dStart;

    CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // top-left 50 x 50 portion: original values, then updated values
    printf("\nOriginal picture (top-left 50 x 50)\n");
    for (int r = 0; r < 50; r++) {
        for (int col = 0; col < 50; col++)
            printf("%3d ", h_in[r * WIDTH + col]);
        printf("\n");
    }
    printf("\nUpdated picture (top-left 50 x 50)\n");
    for (int r = 0; r < 50; r++) {
        for (int col = 0; col < 50; col++)
            printf("%3d ", h_out[r * WIDTH + col]);
        printf("\n");
    }

    // check every pixel on the CPU: expected = 2 x original, capped at 255
    int mismatches = 0, capped = 0;
    for (size_t i = 0; i < count; i++) {
        int expected = 2 * h_in[i];
        if (expected > 255) {
            expected = 255;
            capped++;
        }
        if (h_out[i] != expected) mismatches++;
    }
    printf("\nVerification: %d mismatches out of %zu\n", mismatches, count);
    printf("Pixels capped at 255: %d\n", capped);
    printf("Time taken - %.7f\n", dElaps);

    free(h_in); free(h_out);
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    return 0;
}
