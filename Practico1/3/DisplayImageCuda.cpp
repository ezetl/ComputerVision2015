#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <chrono>

#include "kernel.h"

typedef std::chrono::microseconds Useconds;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        std::cout << "usage: DisplayImage <Image_Path>" << std::endl;
        return -1;
    }

    cv::Mat image;
    image = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR);

    if ( !image.data )
    {
        std::cout << "No image data" << std::endl;
        return -1;
    }

    cv::Mat gray(image.rows, image.cols, CV_8UC1);
    Byte* grayPointer = gray.ptr<Byte>();

    for (unsigned int i = 0U; i < image.rows; ++i)
        for (unsigned int j = 0U; j < image.cols; ++j)
        {
            cv::Point3_<Byte>* p = image.ptr<cv::Point3_<Byte> >(i,j);
            grayPointer[i * image.cols + j] = 0.2126f * p->x +
                                              0.7152f * p->y +
                                              0.0722f * p->z;
        }

    std::vector<float> gaussianKernel ({
                                       0.045f, 0.18f, 0.045f,
                                       0.18f, 0.10f, 0.18f,
                                       0.045f, 0.18f, 0.045f
                                      });


    cv::Mat gaussianBlurCUDA(gray.rows, gray.cols, CV_8UC1);
    Byte* gaussianPointer = gaussianBlurCUDA.ptr<Byte>();

    gaussianBlurCuda(grayPointer,
                     gaussianPointer,
                     gray.cols,
                     gray.rows,
                     &(gaussianKernel[0]));

    cv::Mat gaussianBlur(gray.rows, gray.cols, CV_8UC1);
    std::vector<Byte*> inputs(gaussianKernel.size(), nullptr);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    for (unsigned int i = 1U; i < (gray.rows - 1); ++i)
        for (unsigned int j = 1U; j < (gray.cols - 1); ++j)
        {
            inputs[0]  = gray.ptr<Byte>((i - 1), (j - 1));
            inputs[1]  = gray.ptr<Byte>((i - 1), j);
            inputs[2]  = gray.ptr<Byte>((i - 1), (j + 1));
            inputs[3]  = gray.ptr<Byte>(i, (j - 1));
            inputs[4]  = gray.ptr<Byte>(i, j);
            inputs[5]  = gray.ptr<Byte>(i, (j + 1));
            inputs[6]  = gray.ptr<Byte>((i + 1), (j - 1));
            inputs[7]  = gray.ptr<Byte>((i + 1), j);
            inputs[8]  = gray.ptr<Byte>((i + 1), (j + 1));

            Byte* output = gaussianBlur.ptr<Byte>(i, j);

            unsigned int tempValue = 0U;
            for (unsigned int it = 0U; it < gaussianKernel.size(); ++it)
                  tempValue += (*(inputs[it])) * gaussianKernel[it];

            *output = (tempValue > 255U)?255U:tempValue;
        }

        Useconds executionTime = std::chrono::duration_cast<Useconds>(
                             std::chrono::system_clock::now() - start);

        std::cout<< "Duracion en CPU: " << (executionTime.count() / 1000) << " ms" << std::endl;


    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);

    cv::namedWindow("Gray Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Gray Image", gray);

    cv::namedWindow("Gaussian CUDA Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Gaussian CUDA Image", gaussianBlurCUDA);

    cv::namedWindow("Gaussian Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Gaussian Image", gaussianBlur);


    cv::waitKey(0);

    return 0;
}
