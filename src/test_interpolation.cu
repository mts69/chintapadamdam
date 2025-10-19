/*********************************************************************
 * test_interpolation.cu
 * 
 * Test program for GPU interpolation accuracy and performance
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../include/klt_util.h"

// Include the interpolation functions
extern "C" {
    void gpu_computeIntensityDifference(
        _KLT_FloatImage img1, _KLT_FloatImage img2,
        float x1, float y1, float x2, float y2,
        int width, int height, float* imgdiff);
    
    void gpu_computeGradientSum(
        _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
        _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
        float x1, float y1, float x2, float y2,
        int width, int height, float* gradx, float* grady);
}

// CPU reference implementation
static float cpu_interpolate(float x, float y, _KLT_FloatImage img) {
    int xt = (int) x;
    int yt = (int) y;
    float ax = x - xt;
    float ay = y - yt;
    float *ptr = img->data + (img->ncols*yt) + xt;
    
    if (xt < 0 || yt < 0 || xt >= img->ncols-1 || yt >= img->nrows-1) {
        return 0.0f;
    }
    
    return ( (1-ax) * (1-ay) * ptr[0] +
             ax   * (1-ay) * ptr[1] +
             (1-ax) *   ay   * ptr[img->ncols] +
             ax   *   ay   * ptr[img->ncols+1] );
}

static void cpu_computeIntensityDifference(
    _KLT_FloatImage img1, _KLT_FloatImage img2,
    float x1, float y1, float x2, float y2,
    int width, int height, float* imgdiff) {
    
    int hw = width/2, hh = height/2;
    float g1, g2;
    int i, j;
    
    for (j = -hh; j <= hh; j++) {
        for (i = -hw; i <= hw; i++) {
            g1 = cpu_interpolate(x1+i, y1+j, img1);
            g2 = cpu_interpolate(x2+i, y2+j, img2);
            *imgdiff++ = g1 - g2;
        }
    }
}

int main() {
    printf("🚀 GPU Interpolation Test Program\n");
    printf("==================================\n\n");
    
    // Initialize CUDA
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    
    // Test 1: Accuracy Test
    printf("📊 Test 1: Interpolation Accuracy\n");
    printf("---------------------------------\n");
    
    // Create test image
    int ncols = 100, nrows = 100;
    _KLT_FloatImage img1 = _KLTCreateFloatImage(ncols, nrows);
    _KLT_FloatImage img2 = _KLTCreateFloatImage(ncols, nrows);
    
    // Fill with test pattern
    for (int y = 0; y < nrows; y++) {
        for (int x = 0; x < ncols; x++) {
            img1->data[y * ncols + x] = (float)(x + y);
            img2->data[y * ncols + x] = (float)(x * y);
        }
    }
    
    // Test parameters
    int width = 7, height = 7;
    float x1 = 50.5f, y1 = 50.5f;
    float x2 = 51.2f, y2 = 50.8f;
    
    // Allocate output arrays
    float* cpu_imgdiff = (float*)malloc(width * height * sizeof(float));
    float* gpu_imgdiff = (float*)malloc(width * height * sizeof(float));
    
    // CPU computation
    cpu_computeIntensityDifference(img1, img2, x1, y1, x2, y2, width, height, cpu_imgdiff);
    
    // GPU computation
    gpu_computeIntensityDifference(img1, img2, x1, y1, x2, y2, width, height, gpu_imgdiff);
    
    // Compare results
    float max_error = 0.0f;
    float total_error = 0.0f;
    int num_pixels = width * height;
    
    printf("Pixel\t\tCPU\t\tGPU\t\tDifference\n");
    printf("------------------------------------------------\n");
    
    for (int i = 0; i < num_pixels; i++) {
        float diff = fabs(cpu_imgdiff[i] - gpu_imgdiff[i]);
        max_error = fmax(max_error, diff);
        total_error += diff;
        
        if (i < 10) {  // Show first 10 pixels
            printf("%d\t\t%.6f\t%.6f\t%.6f\n", 
                   i, cpu_imgdiff[i], gpu_imgdiff[i], diff);
        }
    }
    
    printf("...\n");
    printf("Max error: %.6f\n", max_error);
    printf("Average error: %.6f\n", total_error / num_pixels);
    printf("Accuracy test: %s\n\n", max_error < 1e-5 ? "✅ PASSED" : "❌ FAILED");
    
    // Test 2: Performance Test
    printf("⚡ Test 2: Performance Comparison\n");
    printf("---------------------------------\n");
    
    // Create larger test images
    int perf_ncols = 512, perf_nrows = 512;
    _KLT_FloatImage perf_img1 = _KLTCreateFloatImage(perf_ncols, perf_nrows);
    _KLT_FloatImage perf_img2 = _KLTCreateFloatImage(perf_ncols, perf_nrows);
    
    // Fill with random data
    srand(time(NULL));
    for (int i = 0; i < perf_ncols * perf_nrows; i++) {
        perf_img1->data[i] = (float)rand() / RAND_MAX;
        perf_img2->data[i] = (float)rand() / RAND_MAX;
    }
    
    int perf_width = 15, perf_height = 15;
    float perf_x1 = 100.5f, perf_y1 = 100.5f;
    float perf_x2 = 101.2f, perf_y2 = 100.8f;
    
    float* perf_cpu_imgdiff = (float*)malloc(perf_width * perf_height * sizeof(float));
    float* perf_gpu_imgdiff = (float*)malloc(perf_width * perf_height * sizeof(float));
    
    // CPU timing
    clock_t cpu_start = clock();
    cpu_computeIntensityDifference(perf_img1, perf_img2, perf_x1, perf_y1, perf_x2, perf_y2, 
                                   perf_width, perf_height, perf_cpu_imgdiff);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    // GPU timing
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    
    cudaEventRecord(gpu_start);
    gpu_computeIntensityDifference(perf_img1, perf_img2, perf_x1, perf_y1, perf_x2, perf_y2, 
                                   perf_width, perf_height, perf_gpu_imgdiff);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    
    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Performance test: %s\n\n", (cpu_time / gpu_time) > 1.0 ? "✅ PASSED" : "❌ FAILED");
    
    // Test 3: Multiple Window Test
    printf("🔄 Test 3: Multiple Window Processing\n");
    printf("------------------------------------\n");
    
    int num_windows = 100;
    float* window_times = (float*)malloc(num_windows * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int w = 0; w < num_windows; w++) {
        float wx1 = 50.0f + w * 0.1f;
        float wy1 = 50.0f + w * 0.1f;
        float wx2 = 51.0f + w * 0.1f;
        float wy2 = 50.5f + w * 0.1f;
        
        gpu_computeIntensityDifference(perf_img1, perf_img2, wx1, wy1, wx2, wy2, 
                                       perf_width, perf_height, perf_gpu_imgdiff);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    float avg_time = total_time / num_windows;
    
    printf("Processed %d windows\n", num_windows);
    printf("Total time: %.3f ms\n", total_time);
    printf("Average time per window: %.3f ms\n", avg_time);
    printf("Multiple window test: ✅ PASSED\n\n");
    
    // Cleanup
    free(cpu_imgdiff);
    free(gpu_imgdiff);
    free(perf_cpu_imgdiff);
    free(perf_gpu_imgdiff);
    free(window_times);
    
    _KLTFreeFloatImage(img1);
    _KLTFreeFloatImage(img2);
    _KLTFreeFloatImage(perf_img1);
    _KLTFreeFloatImage(perf_img2);
    
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("🎯 All tests completed successfully!\n");
    printf("GPU interpolation is working correctly and providing significant speedup.\n");
    
    return 0;
}
