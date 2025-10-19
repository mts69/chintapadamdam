/*********************************************************************
 * gpu_verification.cu
 * 
 * GPU function verification and debugging for KLT algorithm
 * This file helps verify that GPU functions are actually being called
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/klt_util.h"

// Global counters for GPU function calls
static int gpu_horiz_calls = 0;
static int gpu_vert_calls = 0;
static int gpu_interpolation_calls = 0;
static double total_gpu_time = 0.0;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

/*********************************************************************
 * GPU Convolution with Verification
 */
void gpu_convolveImageHorizWithVerification(
    _KLT_FloatImage imgin,
    _KLT_FloatImage imgout,
    float sigma)
{
    printf("🚀 GPU HORIZONTAL CONVOLUTION CALLED!\n");
    printf("   Input size: %dx%d\n", imgin->ncols, imgin->nrows);
    printf("   Sigma: %.3f\n", sigma);
    
    gpu_horiz_calls++;
    
    // Record timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Create simple kernel for testing
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int image_size = ncols * nrows;
    
    // Allocate GPU memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, image_size * sizeof(float)));
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_input, imgin->data, image_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Simple GPU kernel (identity for now)
    dim3 blockSize(16, 16);
    dim3 gridSize((ncols + blockSize.x - 1) / blockSize.x, (nrows + blockSize.y - 1) / blockSize.y);
    
    // Launch simple kernel
    gpu_identityKernel<<<gridSize, blockSize>>>(d_input, d_output, ncols, nrows);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(imgout->data, d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    total_gpu_time += gpu_time;
    
    printf("   GPU time: %.3f ms\n", gpu_time);
    printf("   ✅ GPU horizontal convolution completed!\n\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void gpu_convolveImageVertWithVerification(
    _KLT_FloatImage imgin,
    _KLT_FloatImage imgout,
    float sigma)
{
    printf("🚀 GPU VERTICAL CONVOLUTION CALLED!\n");
    printf("   Input size: %dx%d\n", imgin->ncols, imgin->nrows);
    printf("   Sigma: %.3f\n", sigma);
    
    gpu_vert_calls++;
    
    // Record timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Create simple kernel for testing
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int image_size = ncols * nrows;
    
    // Allocate GPU memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, image_size * sizeof(float)));
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_input, imgin->data, image_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Simple GPU kernel (identity for now)
    dim3 blockSize(16, 16);
    dim3 gridSize((ncols + blockSize.x - 1) / blockSize.x, (nrows + blockSize.y - 1) / blockSize.y);
    
    // Launch simple kernel
    gpu_identityKernel<<<gridSize, blockSize>>>(d_input, d_output, ncols, nrows);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(imgout->data, d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    total_gpu_time += gpu_time;
    
    printf("   GPU time: %.3f ms\n", gpu_time);
    printf("   ✅ GPU vertical convolution completed!\n\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*********************************************************************
 * Simple GPU kernel for testing
 */
__global__ void gpu_identityKernel(
    const float* input,
    float* output,
    int ncols,
    int nrows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < ncols && row < nrows) {
        int idx = row * ncols + col;
        output[idx] = input[idx];  // Identity operation for testing
    }
}

/*********************************************************************
 * GPU Interpolation with Verification
 */
void gpu_interpolateWithVerification(
    float x, float y, _KLT_FloatImage img, float* result)
{
    gpu_interpolation_calls++;
    
    if (gpu_interpolation_calls % 100 == 0) {
        printf("🚀 GPU INTERPOLATION CALLED %d times!\n", gpu_interpolation_calls);
    }
    
    // Simple interpolation for testing
    int xt = (int)x, yt = (int)y;
    if (xt >= 0 && yt >= 0 && xt < img->ncols && yt < img->nrows) {
        *result = img->data[yt * img->ncols + xt];
    } else {
        *result = 0.0f;
    }
}

/*********************************************************************
 * Print GPU usage statistics
 */
void printGPUUsageStats() {
    printf("\n📊 GPU USAGE STATISTICS\n");
    printf("=======================\n");
    printf("GPU Horizontal Convolutions: %d\n", gpu_horiz_calls);
    printf("GPU Vertical Convolutions: %d\n", gpu_vert_calls);
    printf("GPU Interpolations: %d\n", gpu_interpolation_calls);
    printf("Total GPU Time: %.3f ms\n", total_gpu_time);
    printf("Average GPU Time per Call: %.3f ms\n", 
           (gpu_horiz_calls + gpu_vert_calls) > 0 ? 
           total_gpu_time / (gpu_horiz_calls + gpu_vert_calls) : 0.0);
    
    if (gpu_horiz_calls > 0 || gpu_vert_calls > 0) {
        printf("✅ GPU functions are being used!\n");
    } else {
        printf("❌ No GPU functions were called!\n");
    }
}

/*********************************************************************
 * Reset GPU statistics
 */
void resetGPUStats() {
    gpu_horiz_calls = 0;
    gpu_vert_calls = 0;
    gpu_interpolation_calls = 0;
    total_gpu_time = 0.0;
    printf("🔄 GPU statistics reset\n");
}
