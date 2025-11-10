#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int phase = 0; phase < (N + TILE - 1) / TILE; ++phase) {
        int tiledRow = row;
        int tiledCol = phase * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (tiledRow < N && tiledCol < N) ? A[tiledRow * N + tiledCol] : 0.0f;

        tiledRow = phase * TILE + threadIdx.y;
        tiledCol = col;
        Bs[threadIdx.y][threadIdx.x] =
            (tiledRow < N && tiledCol < N) ? B[tiledRow * N + tiledCol] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

double matmul_cpu(const float* A, const float* B, float* C, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

void write_matrix(std::ofstream& ofs, const char* name, const float* M, int N) {
    ofs << name << " (" << N << "x" << N << ")\n";
    ofs << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ofs << std::setw(12) << M[i * N + j];
        }
        ofs << '\n';
    }
    ofs << '\n';
}

int main() {
    const int N = 512;
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    float *hA = static_cast<float*>(std::malloc(bytes));
    float *hB = static_cast<float*>(std::malloc(bytes));
    float *hC_gpu = static_cast<float*>(std::malloc(bytes));
    float *hC_cpu = static_cast<float*>(std::malloc(bytes));

    if (!hA || !hB || !hC_gpu || !hC_cpu) {
        std::fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
        hC_gpu[i] = 0.0f;
        hC_cpu[i] = 0.0f;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, bytes);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::printf("GPU kernel time: %.3f ms\n", elapsed_ms);

    cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost);

    double cpu_ms = matmul_cpu(hA, hB, hC_cpu, N);
    std::printf("CPU matmul time: %.3f ms\n", cpu_ms);

    bool ok = true;
    for (int i = 0; i < N * N; ++i) {
        float expected = 2.0f * N;
        if (std::fabs(hC_cpu[i] - expected) > 1e-3f) {
            std::printf("CPU mismatch at index %d: got %.6f expected %.6f\n", i, hC_cpu[i], expected);
            ok = false;
            break;
        }
    }

    float max_abs_diff = 0.0f;
    if (ok) {
        for (int i = 0; i < N * N; ++i) {
            float diff = std::fabs(hC_gpu[i] - hC_cpu[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff > 1e-2f) {
                std::printf("GPU/CPU mismatch at index %d: gpu %.6f cpu %.6f\n", i, hC_gpu[i], hC_cpu[i]);
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        std::printf("Max |GPU-CPU| diff: %.6f\n", max_abs_diff);
    } else {
        std::printf("Max |GPU-CPU| diff (up to failure): %.6f\n", max_abs_diff);
    }

    std::ofstream ofs("matrix_log.txt", std::ios::out | std::ios::trunc);
    if (ofs) {
        ofs << "N=" << N << "\n";
        ofs << "GPU time (ms): " << elapsed_ms << "\n";
        ofs << "CPU time (ms): " << cpu_ms << "\n";
        ofs << "Max |GPU-CPU| diff: " << max_abs_diff << "\n\n";
        write_matrix(ofs, "Matrix A", hA, N);
        write_matrix(ofs, "Matrix B", hB, N);
        write_matrix(ofs, "CPU Result", hC_cpu, N);
        write_matrix(ofs, "GPU Result", hC_gpu, N);
    } else {
        std::fprintf(stderr, "Failed to open matrix_log.txt for writing\n");
    }

    std::printf("Result %s\n", ok ? "OK" : "FAILED");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    std::free(hA);
    std::free(hB);
    std::free(hC_cpu);
    std::free(hC_gpu);
    return ok ? 0 : 1;
}
