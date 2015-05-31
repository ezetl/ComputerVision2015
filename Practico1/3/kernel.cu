
#include "kernel.h"

__global__ void gaussianBlurKernel(const Byte* const __restrict__ input,
                                   Byte* const __restrict__ output,
                                   const size_t width,
                                   const size_t height,
                                   const float* const __restrict__ gaussianKernel)
{
//x and y maxs are width and height
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    Byte inputs[9];

    if((x > 0) && (x < (height - 1)) && (y > 0) && (y < (width - 1)))
    {
        inputs[0]  = input[(y - 1) * width + (x - 1)];
        inputs[1]  = input[(y - 1) * width + x];
        inputs[2]  = input[(y - 1) * width +  (x + 1)];
        inputs[3]  = input[y * width + (x - 1)];
        inputs[4]  = input[y * width + x];
        inputs[5]  = input[y * width + (x + 1)];
        inputs[6]  = input[(y + 1) * width + (x - 1)];
        inputs[7]  = input[(y + 1) * width + x];
        inputs[8]  = input[(y + 1) * width + (x + 1)];

        unsigned int tempValue = 0;
        for (unsigned int it = 0; it < 9; ++it)
              tempValue += inputs[it] * gaussianKernel[it];

        output[y * width + x] = (tempValue > 255)?255:tempValue;
    }
    else
        output[y * width + x] = 255;

};


void gaussianBlurCuda(const Byte* const input,
                      Byte* const output,
                      const size_t width,
                      const size_t height,
                      const float* const gaussianKernel)
{
    Byte* cudaInput;
    Byte* cudaOutput;
    float* cudaKernel;
    cudaMalloc(reinterpret_cast<void**>(&cudaInput), width * height * sizeof(Byte));
    cudaMalloc(reinterpret_cast<void**>(&cudaOutput), width * height * sizeof(Byte));
    cudaMalloc(reinterpret_cast<void**>(&cudaKernel), 9 * sizeof(float));

    cudaMemcpy(cudaInput, input, width * height * sizeof(Byte), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaKernel, gaussianKernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(width / BLOCK_SIZE_X, height / BLOCK_SIZE_Y);
   
    gaussianBlurKernel<<<gridSize, blockSize>>>(cudaInput,
                                                cudaOutput,
                                                width,
                                                height,
                                                cudaKernel);

    cudaDeviceSynchronize();
    cudaMemcpy(output, cudaOutput, width * height * sizeof(Byte), cudaMemcpyDeviceToHost);

    cudaFree(cudaInput);
    cudaFree(cudaOutput);
    cudaFree(cudaKernel);
}
