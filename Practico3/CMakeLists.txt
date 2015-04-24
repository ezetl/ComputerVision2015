cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED)
INCLUDE_DIRECTORIES(
                    ${OpenCV_INCLUDE_DIRS}
                    /usr/local/cuda/include
                    ${CUDA_BIN_PATH}/include
                    ${CUDA_BIN_PATH}/lib
                   )
link_directories( /usr/lib64 )
link_directories( /usr/local/cuda/lib64 )

SET(CUDA_64_BIT_DEVICE_CODE_DEFAULT ON)
SET(CUDA_NVCC_FLAGS "-L/usr/lib64 -L/usr/local/cuda/lib64 -lcuda -lcudart --ptxas-options=-v")

message ("cuda path: ${CUDA_BIN_PATH}")

add_definitions(-std=c++0x)
cuda_add_library(HarrisKernel
                 ./kernel.cu
                )

add_executable(DisplayImage DisplayImageCuda.cpp)
ADD_DEPENDENCIES(DisplayImage HarrisKernel)
target_link_libraries(DisplayImage
                      ${OpenCV_LIBS}
                      HarrisKernel
                      cuda
                     )
