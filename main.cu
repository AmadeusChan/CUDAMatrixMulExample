#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <utility>

#include "omp.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define CHECK_CUDA(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// matrix M x K times K x N => M x N
#define SIZE_M (4 * 1024)
#define SIZE_K (4 * 1024)
#define SIZE_N (4 * 1024)
#define RUNS (32)

// 32 x 32 x 4 * 3 = 12KB
// could be larger 
#define TILE_X (128)
#define TILE_Y (128)
#define TILE_Z (32)

#define WARP_SIZE (32)
#define NUM_THREADS (1024)

void init_data(float * data, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++ i) {
        data[i] = rand() % 100;
    }
}

std::pair<float*, float*> prepare_matrix(int rows, int cols) {
    size_t num_elements = (size_t) rows * cols;
    float * data = NULL;
    CHECK_CUDA(cudaMalloc(&data, sizeof(float) * num_elements));
    assert(data);
    // allocate on CPU
    float * data_cpu = NULL;
    CHECK_CUDA(cudaMallocHost(&data_cpu, sizeof(float) * num_elements));
    init_data(data_cpu, num_elements);
    // copy the data to GPU
    CHECK_CUDA(cudaMemcpy(
                data, data_cpu, sizeof(float) * num_elements,
                cudaMemcpyHostToDevice
                ));
    return std::make_pair(data, data_cpu);
}

void dealloc_matrix(float * data, float * data_cpu) {
    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaFreeHost(data_cpu));
}

void verify(
        float * mat_a_cpu, float * mat_b_cpu, float * mat_c_cpu,
        float * mat_c_gpu,
        int m, int k, int n
        ) {
    CHECK_CUDA(cudaMemcpy(
                mat_c_cpu, mat_c_gpu, sizeof(float) * m * n,
                cudaMemcpyDeviceToHost
                ));
    printf("Waiting for verification...\n");
    // matA: m x k 
    // matB: k x n
    bool passed = true;
    int tests = 10000; // 10K
    for (int test = 0; test < tests; ++ test) {
        // otherwise it is tooooo slow
        int i = rand() % m;
        int j = rand() % n;
        // calculate the groundtruth
        float sum = 0.;
        for (int w = 0; w < k; ++ w) {
            sum += mat_a_cpu[i * k + w] * mat_b_cpu[w * n + j];
        }
        if (abs(sum - mat_c_cpu[i * n + j]) / abs(mat_c_cpu[i * n + j]) > 1e-2) {
            passed = false;
            break;
        }
    }
    if (! passed) {
        fprintf(stderr, "ERROR FOUND!\n");
    } else {
        fprintf(stderr, "Correctness verified.\n");
    }
}

template<int BLOCK_SIZE>
__global__ void matmul_kernel(
        const float * __restrict__ mat_a,
        const float * __restrict__ mat_b,
        float * __restrict__ mat_c,
        const int m, const int k, const int n
        ) {
    // tile (tile_x, tile_y) will be processed by this thread block
    const int tile_x = blockIdx.x; 
    const int tile_y = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    assert(BLOCK_SIZE % WARP_SIZE == 0);

    // TODO: fixed potential bank conflict issues if applicable
    extern __shared__ float sm[];

    float * tile_a = &sm[0];
    float * tile_b = &tile_a[TILE_X * TILE_Z];
    float * tile_c = &tile_b[TILE_Z * TILE_Y];

    // set the tile_c to zeros
    for (int i = warp; i < TILE_X; i += NUM_WARPS) {
        for (int j = lane; j < TILE_Y; j += WARP_SIZE) {
            tile_c[i * TILE_Y + j] = 0.0f;
        }
    }

    // determine how to parallelize within the thread block
    const int num_tiles_z = k / TILE_Z;
    const int x = tile_x * TILE_X;
    const int y = tile_y * TILE_Y;

    for (int tile_z = 0; tile_z < num_tiles_z; ++ tile_z) {
        // tile (tile_x, tile_z) of matrix A 
        // will be multiplied with (tile_z, tile_y) of 
        // matrix B, and the results will be written to 
        // tile (tile_x, tile_y) of matrix C

        // firstly, load the tiles of matrix A and B to the 
        // shared memory
        const int z = tile_z * TILE_Z;

        //for (int i = warp; i < TILE_X; i += NUM_WARPS) {
        //    for (int j = lane; j < TILE_Z; j += WARP_SIZE) {
        //        tile_a[i * TILE_Z + j] = mat_a[(x + i) * k + (z + j)];
        //    }
        //}
        for (int i = warp; i < TILE_Z; i += NUM_WARPS) {
            for (int j = lane; j < TILE_Y; j += WARP_SIZE) {
                tile_b[i * TILE_Y + j] = mat_b[(z + i) * n + (y + j)];
            }
        }
        __syncthreads();

        for (int i = warp; i < TILE_X; i += NUM_WARPS) {
            // load the data
            for (int j = lane; j < TILE_Z; j += WARP_SIZE) {
                tile_a[i * TILE_Z + j] = mat_a[(x + i) * k + (z + j)];
            }
            __syncwarp();

            // process the loaded data
            for (int j = lane * 4; j < TILE_Y; j += WARP_SIZE * 4) {
                float4 sum;
                sum.x = sum.y = sum.z = sum.w = 0.0f;
#pragma unroll
                for (int w = 0; w < TILE_Z; w += 4) {
                    // shared memory latency is HIGHHHHH => TRIES TO AVOID THEM
                    float4 a = *reinterpret_cast<float4*>(&tile_a[i * TILE_Z + w]);
                    float4 b1 = *reinterpret_cast<float4*>(&tile_b[w * TILE_Y + j]);
                    float4 b2 = *reinterpret_cast<float4*>(&tile_b[(w + 1) * TILE_Y + j]);
                    float4 b3 = *reinterpret_cast<float4*>(&tile_b[(w + 2) * TILE_Y + j]);
                    float4 b4 = *reinterpret_cast<float4*>(&tile_b[(w + 3) * TILE_Y + j]);

                    // a.x
                    sum.x += a.x * b1.x;
                    sum.y += a.x * b1.y;
                    sum.z += a.x * b1.z;
                    sum.w += a.x * b1.w;
                    // a.y
                    sum.x += a.y * b2.x;
                    sum.y += a.y * b2.y;
                    sum.z += a.y * b2.z;
                    sum.w += a.y * b2.w;
                    // a.z
                    sum.x += a.z * b3.x;
                    sum.y += a.z * b3.y;
                    sum.z += a.z * b3.z;
                    sum.w += a.z * b3.w;
                    // a.w
                    sum.x += a.w * b4.x;
                    sum.y += a.w * b4.y;
                    sum.z += a.w * b4.z;
                    sum.w += a.w * b4.w;
                }
                float4 dst = *reinterpret_cast<float4*>(&tile_c[i * TILE_Y + j]);
                dst.x += sum.x; 
                dst.y += sum.y;
                dst.z += sum.z;
                dst.w += sum.w;
                *reinterpret_cast<float4*>(&tile_c[i * TILE_Y + j]) = dst;
            }
        }
        
        __syncthreads();
    }

    // write back the result to the global memory of the result matrix
    for (int i = warp; i < TILE_X; i += NUM_WARPS) {
        for (int j = lane; j < TILE_Y; j += WARP_SIZE) {
            mat_c[(x + i) * n + (y + j)] = tile_c[i * TILE_Y + j];
        }
    }
}

void matrix_mul_gpu(
        float * mat_a, float * mat_b, float * mat_c,
        int m, int k, int n
        ) {
    // we simply let each thread block produce a tile in the 
    // result matrix
    dim3 grid(m / TILE_X, n / TILE_Y);
    assert(m % TILE_X == 0 && n % TILE_Y == 0);

    int maxbytes = sizeof(float) * (TILE_X * TILE_Y + TILE_X * TILE_Z + TILE_Z * TILE_Y);
    CHECK_CUDA(
            cudaFuncSetAttribute(
                matmul_kernel<NUM_THREADS>, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, 
                maxbytes
                )
            );

    // invoke the GPU kernel
    matmul_kernel<NUM_THREADS><<<grid, NUM_THREADS, maxbytes>>>(
            mat_a, mat_b, mat_c, m, k, n
            );
}

int main(int argc, char ** argv) {
    CHECK_CUDA(cudaSetDevice(1));

    srand(17);
    // prepare the input matrix
    std::pair<float*, float*> mat_a = prepare_matrix(SIZE_M, SIZE_K);
    std::pair<float*, float*> mat_b = prepare_matrix(SIZE_K, SIZE_N);
    std::pair<float*, float*> mat_c = prepare_matrix(SIZE_M, SIZE_N);

    // warmming up
    printf("Warming up\n");
    matrix_mul_gpu(
            mat_a.first, mat_b.first, mat_c.first,
            SIZE_M, SIZE_K, SIZE_N
            );
    verify( 
            mat_a.second, mat_b.second, mat_c.second, mat_c.first,
            SIZE_M, SIZE_K, SIZE_N
            );

    printf("Start Benchmarking...\n");
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&end_event));

    CHECK_CUDA(cudaEventRecord(start_event));
    for (int run = 0; run < RUNS; ++ run) {
        matrix_mul_gpu(
                mat_a.first, mat_b.first, mat_c.first,
                SIZE_M, SIZE_K, SIZE_N
                );
    }
    CHECK_CUDA(cudaEventRecord(end_event));
    CHECK_CUDA(cudaEventSynchronize(end_event));

    float runtime = 0.;
    CHECK_CUDA(cudaEventElapsedTime(&runtime, start_event, end_event));
    runtime /= 1e3;
    float flop = RUNS * 2. * SIZE_M * SIZE_K * SIZE_N;
    float gflops = flop / runtime / 1e9;
    printf("The FLOPS achieved is %.3f GFLOPS\n", gflops);

    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(end_event));

    dealloc_matrix(mat_a.first, mat_a.second);
    dealloc_matrix(mat_b.first, mat_b.second);
    dealloc_matrix(mat_c.first, mat_c.second);

    return 0;
}
