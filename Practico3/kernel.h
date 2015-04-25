#ifndef GAUSSIAN_KERNEL_H
#define GAUSSIAN_KERNEL_H
#define THRESHOLD 20000.0

typedef unsigned char Byte;

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8


    void gaussianBlurCuda(const float* const input,
                          float* const output);
    void gradientSobelCuda();
    void calculateA();
    void calculateR();
    void threshold();
    void nonMaximaSupression();
    void harrisCornersFilter(const float* const image,
                             const size_t imageWidth,
                             const size_t imageHeight,
                             const float* const gaussianKernel,
                             float* output,
                             int* features);


#endif  //UCI_CUDAKERNEL_H
