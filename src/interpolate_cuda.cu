#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "../include/klt_util.h"

/*********************************************************************
 * GPU Bilinear Interpolation Kernel
 */
__device__ __forceinline__ float gpu_interpolate(
    float x, 
    float y, 
    const float* img_data,
    int ncols, 
    int nrows)
{
    // Extract integer and fractional parts
    int xt = (int) x;
    int yt = (int) y;
    float ax = x - xt;
    float ay = y - yt;
    
    // Bounds checking
    if (xt < 0 || yt < 0 || xt >= ncols - 1 || yt >= nrows - 1) {
        return 0.0f;
    }
    
    // Calculate pointer offset
    const float* ptr = img_data + (ncols * yt) + xt;
    
    // Bilinear interpolation
    float val = (1.0f - ax) * (1.0f - ay) * ptr[0] +
                ax * (1.0f - ay) * ptr[1] +
                (1.0f - ax) * ay * ptr[ncols] +
                ax * ay * ptr[ncols + 1];
    
    return val;
}

/*********************************************************************
 * GPU Intensity Difference Kernel
 */
__global__ void gpu_computeIntensityDifferenceKernel(
    const float* img1_data, int img1_ncols, int img1_nrows,
    const float* img2_data, int img2_ncols, int img2_nrows,
    float x1, float y1, float x2, float y2,
    int width, int height, float* imgdiff)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    float x1_val = x1 + col - width/2.0f;
    float y1_val = y1 + row - height/2.0f;
    float x2_val = x2 + col - width/2.0f;
    float y2_val = y2 + row - height/2.0f;
    
    float val1 = gpu_interpolate(x1_val, y1_val, img1_data, img1_ncols, img1_nrows);
    float val2 = gpu_interpolate(x2_val, y2_val, img2_data, img2_ncols, img2_nrows);
    
    imgdiff[row * width + col] = val1 - val2;
}

/*********************************************************************
 * GPU Gradient Sum Kernel
 */
__global__ void gpu_computeGradientSumKernel(
    const float* gradx1_data, int gradx1_ncols, int gradx1_nrows,
    const float* grady1_data, int grady1_ncols, int grady1_nrows,
    const float* gradx2_data, int gradx2_ncols, int gradx2_nrows,
    const float* grady2_data, int grady2_ncols, int grady2_nrows,
    float x1, float y1, float x2, float y2,
    int width, int height, float* gradx, float* grady)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    float x1_val = x1 + col - width/2.0f;
    float y1_val = y1 + row - height/2.0f;
    float x2_val = x2 + col - width/2.0f;
    float y2_val = y2 + row - height/2.0f;
    
    float gx1 = gpu_interpolate(x1_val, y1_val, gradx1_data, gradx1_ncols, gradx1_nrows);
    float gy1 = gpu_interpolate(x1_val, y1_val, grady1_data, grady1_ncols, grady1_nrows);
    float gx2 = gpu_interpolate(x2_val, y2_val, gradx2_data, gradx2_ncols, gradx2_nrows);
    float gy2 = gpu_interpolate(x2_val, y2_val, grady2_data, grady2_ncols, grady2_nrows);
    
    gradx[row * width + col] = gx1 + gx2;
    grady[row * width + col] = gy1 + gy2;
}

/*********************************************************************
 * GPU Intensity Difference Host Function
 */
extern "C" void gpu_computeIntensityDifference(
    _KLT_FloatImage img1, _KLT_FloatImage img2,
    float x1, float y1, float x2, float y2,
    int width, int height, float* imgdiff)
{
    printf("🚀 GPU INTENSITY DIFFERENCE (Size: %dx%d)\n", width, height);
    
    // Use basic CUDA operations
    float *d_img1, *d_img2, *d_imgdiff;
    
    // Allocate GPU memory
    cudaMalloc(&d_img1, img1->ncols * img1->nrows * sizeof(float));
    cudaMalloc(&d_img2, img2->ncols * img2->nrows * sizeof(float));
    cudaMalloc(&d_imgdiff, width * height * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_img1, img1->data, img1->ncols * img1->nrows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2->data, img2->ncols * img2->nrows * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    gpu_computeIntensityDifferenceKernel<<<gridSize, blockSize>>>(
        d_img1, img1->ncols, img1->nrows,
        d_img2, img2->ncols, img2->nrows,
        x1, y1, x2, y2, width, height, d_imgdiff);
    
    // Check for kernel launch errors
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        printf("   ❌ GPU Interpolation kernel launch failed: %s\n", cudaGetErrorString(kernel_error));
        cudaFree(d_img1);
        cudaFree(d_img2);
        cudaFree(d_imgdiff);
        return;
    }
    
    // Wait for kernel to complete
    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        printf("   ❌ GPU Interpolation kernel execution failed: %s\n", cudaGetErrorString(sync_error));
        cudaFree(d_img1);
        cudaFree(d_img2);
        cudaFree(d_imgdiff);
        return;
    }
    
    // Copy result back
    cudaMemcpy(imgdiff, d_imgdiff, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_imgdiff);
}

/*********************************************************************
 * GPU Gradient Sum Host Function
 */
extern "C" void gpu_computeGradientSum(
    _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
    _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
    float x1, float y1, float x2, float y2,
    int width, int height, float* gradx, float* grady)
{
    printf("🚀 GPU GRADIENT SUM (Size: %dx%d)\n", width, height);
    
    // Use basic CUDA operations
    float *d_gradx1, *d_grady1, *d_gradx2, *d_grady2, *d_gradx, *d_grady;
    
    // Allocate GPU memory
    cudaMalloc(&d_gradx1, gradx1->ncols * gradx1->nrows * sizeof(float));
    cudaMalloc(&d_grady1, grady1->ncols * grady1->nrows * sizeof(float));
    cudaMalloc(&d_gradx2, gradx2->ncols * gradx2->nrows * sizeof(float));
    cudaMalloc(&d_grady2, grady2->ncols * grady2->nrows * sizeof(float));
    cudaMalloc(&d_gradx, width * height * sizeof(float));
    cudaMalloc(&d_grady, width * height * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_gradx1, gradx1->data, gradx1->ncols * gradx1->nrows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady1, grady1->data, grady1->ncols * grady1->nrows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradx2, gradx2->data, gradx2->ncols * gradx2->nrows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady2, grady2->data, grady2->ncols * grady2->nrows * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    gpu_computeGradientSumKernel<<<gridSize, blockSize>>>(
        d_gradx1, gradx1->ncols, gradx1->nrows,
        d_grady1, grady1->ncols, grady1->nrows,
        d_gradx2, gradx2->ncols, gradx2->nrows,
        d_grady2, grady2->ncols, grady2->nrows,
        x1, y1, x2, y2, width, height, d_gradx, d_grady);
    
    // Check for kernel launch errors
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        printf("   ❌ GPU Gradient kernel launch failed: %s\n", cudaGetErrorString(kernel_error));
        cudaFree(d_gradx1);
        cudaFree(d_grady1);
        cudaFree(d_gradx2);
        cudaFree(d_grady2);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        return;
    }
    
    // Wait for kernel to complete
    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        printf("   ❌ GPU Gradient kernel execution failed: %s\n", cudaGetErrorString(sync_error));
        cudaFree(d_gradx1);
        cudaFree(d_grady1);
        cudaFree(d_gradx2);
        cudaFree(d_grady2);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        return;
    }
    
    // Copy result back
    cudaMemcpy(gradx, d_gradx, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grady, d_grady, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_gradx1);
    cudaFree(d_grady1);
    cudaFree(d_gradx2);
    cudaFree(d_grady2);
    cudaFree(d_gradx);
    cudaFree(d_grady);
}

/*********************************************************************
 * GPU Single Point Interpolation
 */
extern "C" float gpu_interpolate_single(float x, float y, _KLT_FloatImage img) {
    // For single point interpolation, GPU overhead is too high
    // Use optimized CPU implementation instead
    static int call_count = 0;
    call_count++;
    
    // Debug: Print first few calls
    if (call_count <= 5) {
        printf("   🔍 GPU Interpolation call #%d: (%.2f, %.2f)\n", call_count, x, y);
    }
    
    int xt = (int) x;
    int yt = (int) y;
    float ax = x - xt;
    float ay = y - yt;
    
    if (xt < 0 || yt < 0 || xt >= img->ncols - 1 || yt >= img->nrows - 1) {
        if (call_count <= 5) {
            printf("   ⚠️  GPU Interpolation: Out of bounds (%.2f, %.2f)\n", x, y);
        }
        return 0.0f;
    }
    
    const float* ptr = img->data + (img->ncols * yt) + xt;
    float val = (1.0f - ax) * (1.0f - ay) * ptr[0] +
                ax * (1.0f - ay) * ptr[1] +
                (1.0f - ax) * ay * ptr[img->ncols] +
                ax * ay * ptr[img->ncols + 1];
    
    if (call_count <= 5) {
        printf("   ✅ GPU Interpolation result: %.3f\n", val);
    }
    
    return val;
}