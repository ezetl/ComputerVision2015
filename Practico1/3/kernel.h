#ifndef GAUSSIAN_KERNEL_H
#define GAUSSIAN_KERNEL_H

typedef unsigned char Byte;

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8


void gaussianBlurCuda(__restrict__ const Byte* const input,
                      __restrict__ Byte* const output,
                      const size_t width,
                      const size_t height,
                      const float* const gaussianKernel);

#endif  //UCI_CUDAKERNEL_H
