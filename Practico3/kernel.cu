#include <opencv2/opencv.hpp>

#include "kernel.h"

/*
   nvcc -c kernel.cu && 
 */

size_t width;
size_t height;
float* cudaInput;
float* cudaOutputX;
float* cudaOutputY;
float* cudaOutputAux;
float* cudaOutputAux2;
float* cudaOutputAux3;
float* gaussianKernelCuda;
float* cudaSobelX;
float* cudaSobelY;
float* cudaA_X_X;
float* cudaA_X_Y;
float* cudaA_Y_Y;
float* cuda_R;
float* cuda_features;


__global__ void gaussianBlurKernel(const float* const __restrict__ input,
                                   float* const __restrict__ output,
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

__global__ void sobelKernel(const float* const __restrict__ input,
                            float* const __restrict__ outputX,
                            float* const __restrict__ outputY,
                            const size_t width,
                            const size_t height,
                            const float* const __restrict__ sobelKernelX,
                            const float* const __restrict__ sobelKernelY)
{
    //x and y maxs are width and height
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float inputs[9];

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

        float tempValueX = 0.0f;
        float tempValueY = 0.0f;
        for (unsigned int it = 0; it < 9; ++it)
        {
              tempValueX += inputs[it] * sobelKernelX[it];
              tempValueY += inputs[it] * sobelKernelY[it];
        }

        outputX[y * width + x] = tempValueX;
        outputY[y * width + x] = tempValueY;
    }
    else
    {
        outputX[y * width + x] = 0.0f;
        outputY[y * width + x] = 0.0f;
    }
};

__global__ void cwiseProduct(const float* const matrix1,
                             const float* const matrix2,
                             float* const output,
                             const size_t width,
                             const size_t height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if((x > 0) && (x < (height - 1)) && (y > 0) && (y < (width - 1)))
    {
        output[y * width + x] = matrix1[y * width + x] * matrix2[y * width + x];
    }
    else
    {
        output[y * width + x] = 0.0f;
    }

}

__global__ void calculate_k_product(const float * const __restrict__ matrix1,
                                    const float * const __restrict__ matrix2,
                                    const float k,
                                    float * const __restrict__ output,
                                    const size_t width,
                                    const size_t height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x > 0) && (x < (height - 1)) && (y > 0) && (y < (width - 1)))
    {
        float aux = matrix1[y * width + x] + matrix2[y * width + x];
        output[y * width + x] = k * aux * aux; 
    }
    else
    {
        output[y * width + x] = 0.0f;
    }
}

__global__ void calculate_diff(float * const __restrict__ matrix1,
                               const float * const __restrict__ matrix2,
                               const float * const __restrict__ matrix3,
                               const size_t width,
                               const size_t height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x > 0) && (x < (height - 1)) && (y > 0) && (y < (width - 1)))
    {
        matrix1[y * width + x] = matrix1[y * width + x] - matrix2[y * width + x] - matrix3[y * width + x]; 
    }
    else
    {
        matrix1[y * width + x] = 0.0f;
    }
}

__global__ void threshold_cuda(float * const R,
                               const float threshold,
                               const size_t width,
                               const size_t height)
{
    // THRESH_TOZERO
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if((x > 0) && (x < (height - 1)) && (y > 0) && (y < (width - 1)))
    {
      if(R[y * width + x] < threshold)
      {
        R[y * width + x] = 0.0f; 
      }
    }
    else
    {
      R[y * width + x] = 0.0f;
    }
}

__global__ void nonMaximaSupression_cuda(const float * const __restrict__ input,
                                         float * const __restrict__ features,
                                         const size_t width,
                                         const size_t height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    Byte neighbours[8]; //todos menos si mismo

    if((x > 0) && (x < (height - 1)) && (y > 0) && (y < (width - 1)))
    {
        neighbours[0]  = input[(y - 1) * width + (x - 1)];
        neighbours[1]  = input[(y - 1) * width + x];
        neighbours[2]  = input[(y - 1) * width +  (x + 1)];
        neighbours[3]  = input[y * width + (x - 1)];
        neighbours[4]  = input[y * width + (x + 1)];
        neighbours[5]  = input[(y + 1) * width + (x - 1)];
        neighbours[6]  = input[(y + 1) * width + x];
        neighbours[7]  = input[(y + 1) * width + (x + 1)];

        int is_max = 1;
        for (unsigned int it = 0; it < 8 && is_max; ++it)
              is_max = neighbours[it] < input[y * width + x];

        if(is_max){
            features[y * width + x] = input[y * width + x];
        }
        else
        {
            features[y * width + x] = 0.0f;
        }
    }
    else
        features[y * width + x] = 0.0f;
}

__global__ void normalize_R(float * const __restrict__ R,
                            const float max,
                            const float min,
                            const size_t width,
                            const size_t height)
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if((x > 0) && (x < (height - 1)) && (y > 0) && (y < (width - 1)))
  {
      R[y * width + x] = R[y * width + x] * (1.0 / (max - min)) - min / (max - min);
  }
  else
  {
      R[y * width + x] = 0.0f;
  }

}

void harrisCornersFilter(const float* const image,
                         const size_t imageWidth,
                         const size_t imageHeight,
                         const float* const gaussianKernel,
                         float* output)
{
    //Inicializacion de memoria

    width = imageWidth;
    height = imageHeight;

    cudaMalloc(reinterpret_cast<void**>(&cudaInput), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&gaussianKernelCuda), 9 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaSobelX), 9 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaSobelY), 9 * sizeof(float));

    cudaMalloc(reinterpret_cast<void**>(&cudaOutputX), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaOutputY), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaOutputAux), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaOutputAux2), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaOutputAux3), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaA_X_X), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaA_X_Y), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cudaA_Y_Y), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cuda_R), width * height * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&cuda_features), width * height * sizeof(float));

    float sobelKernelX[] = {-1.0f, 0.0f, 1.0f,
                            -2.0f, 0.0f, 2.0f,
                            -1.0f, 0.0f, 1.0f
                           };

    float sobelKernelY[] = {-1.0f, -2.0f, -1.0f,
                             0.0f,  0.0f,  0.0f,
                             1.0f,  2.0f,  1.0f
                           };

    cudaMemcpy(cudaInput, image, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gaussianKernelCuda, gaussianKernel, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaSobelX, sobelKernelX, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaSobelY, sobelKernelY, 9 * sizeof(float), cudaMemcpyHostToDevice);

//Comienzo del calculo

    gradientSobelCuda();
    cudaDeviceSynchronize();

    calculateA();

    //tip: Agregar outputs y usar streams

    gaussianBlurCuda(cudaA_X_X,cudaOutputAux);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaA_X_X, cudaOutputAux, width * height * sizeof(float), cudaMemcpyDeviceToDevice);

    gaussianBlurCuda(cudaA_X_Y,cudaOutputAux);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaA_X_Y, cudaOutputAux, width * height * sizeof(float), cudaMemcpyDeviceToDevice);

    gaussianBlurCuda(cudaA_Y_Y,cudaOutputAux);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaA_Y_Y, cudaOutputAux, width * height * sizeof(float), cudaMemcpyDeviceToDevice);

    calculateR();
    cudaDeviceSynchronize();

    threshold();
    cudaDeviceSynchronize();

    //Aqui dentro dejar en el rango [0, 1] a cada pixel de la imagen;
    nonMaximaSupression();
    cudaDeviceSynchronize();

//copiamos el resultado
    cudaMemcpy(output, cudaOutputAux, width * height * sizeof(float), cudaMemcpyDeviceToHost);

//Liberamos memoria

    cudaFree(cudaInput);
    cudaFree(cudaOutputX);
    cudaFree(cudaOutputY);
    cudaFree(cudaOutputAux);
    cudaFree(cudaOutputAux3);
    cudaFree(cudaOutputAux2);
    cudaFree(gaussianKernelCuda);
    cudaFree(cudaSobelX);
    cudaFree(cudaSobelY);
    cudaFree(cudaA_X_X);
    cudaFree(cudaA_X_Y);
    cudaFree(cudaA_Y_Y);
    cudaFree(cuda_features);
}

void gaussianBlurCuda(const float* const input,
                      float* const output)
{
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(width / BLOCK_SIZE_X, height / BLOCK_SIZE_Y);

    gaussianBlurKernel<<<gridSize, blockSize>>>(input,
                                                output,
                                                width,
                                                height,
                                                gaussianKernelCuda);
}

void gradientSobelCuda()
{
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize(width / BLOCK_SIZE_X, height / BLOCK_SIZE_Y);

    sobelKernel<<<gridSize, blockSize>>>(cudaInput,
                                         cudaOutputX,
                                         cudaOutputY,
                                         width,
                                         height,
                                         cudaSobelX,
                                         cudaSobelY);
}

void calculateA()
{
  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize(width / BLOCK_SIZE_X, height / BLOCK_SIZE_Y);

  //tip: Se pueden usar streams

  //cudaA_X_X = cudaOutputX * cudaOutputX;
  cwiseProduct<<<gridSize, blockSize>>>(cudaOutputX,
                                        cudaOutputX,
                                        cudaA_X_X,
                                        width,
                                        height);
  //cudaA_X_Y = cudaOutputX * cudaOutputY;
  cwiseProduct<<<gridSize, blockSize>>>(cudaOutputX,
                                        cudaOutputY,
                                        cudaA_X_Y,
                                        width,
                                        height);

  //cudaA_Y_Y = cudaOutputY * cudaOutputY;
  cwiseProduct<<<gridSize, blockSize>>>(cudaOutputY,
                                        cudaOutputY,
                                        cudaA_Y_Y,
                                        width,
                                        height);
}

void calculateR()
{
//TODO Ejercicio 2 hacer el kernel para calcular R con la siguiente funcion y llamarlo.
  // Guarda el resultado en  cudaOutputAux
  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize(width / BLOCK_SIZE_X, height / BLOCK_SIZE_Y);
  const float k = 0.04f;

  cwiseProduct<<<gridSize, blockSize>>>(cudaA_X_X,
                                        cudaA_Y_Y,
                                        cudaOutputAux,
                                        width,
                                        height);

  cwiseProduct<<<gridSize, blockSize>>>(cudaA_X_Y,
                                        cudaA_X_Y,
                                        cudaOutputAux2,
                                        width,
                                        height);

  calculate_k_product<<<gridSize, blockSize>>>(cudaA_X_X,
                                               cudaA_Y_Y,
                                               k,
                                               cudaOutputAux3,
                                               width,
                                               height);

  calculate_diff<<<gridSize, blockSize>>>(cudaOutputAux,
                                          cudaOutputAux2,
                                          cudaOutputAux3,
                                          width,
                                          height);
}



void threshold()
{
//TODO Ejercicio 3 calcular el umbral a R (cudaOutputAux)
  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize(width / BLOCK_SIZE_X, height / BLOCK_SIZE_Y);
  float thresh = 100.0;
  threshold_cuda<<<gridSize, blockSize>>>(cudaOutputAux,
                                          thresh,
                                          width,
                                          height);
}



void nonMaximaSupression()
{
  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize(width / BLOCK_SIZE_X, height / BLOCK_SIZE_Y);
  //TODO Ejercicio 4 calcular NMS a R (cudaOutputAux) y dejar R en el rango [0, 1]
  nonMaximaSupression_cuda<<<gridSize, blockSize>>>(cudaOutputAux,
                                               cuda_features,
                                               width,
                                               height);
  //TODO: implementar el reduce en CUDA
  cv::Mat R(width, height, CV_32FC1);
  cudaMemcpy(R.ptr<float>(), cudaOutputAux, width * height * sizeof(float), cudaMemcpyDeviceToHost);
  double min;
  double max;
  cv::minMaxIdx(R, &min, &max);

  normalize_R<<<gridSize, blockSize>>>(cudaOutputAux,
                                       min,
                                       max,
                                       width,
                                       height);
}
