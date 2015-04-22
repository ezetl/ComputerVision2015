#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <chrono>

#include "kernel.h"

typedef std::chrono::microseconds Useconds;

void print_features(cv::Mat& img, cv::Mat& orig)
{
  for (int i = 1; i < img.rows - 1; ++i) {
    float* img_1 = img.ptr<float>(i);
    for (int j = 1; j < img.cols - 1; ++j) {
      if(img_1[j]>0.0f)
        cv::circle(orig, cv::Point(j, i), 5, cv::Scalar(0), 2, 8, 0);
    }
  }
}

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


    // convierte a escala de grises (uint8)
    cv::Mat gray_UINT8;
    cv::cvtColor(image, gray_UINT8, CV_BGR2GRAY);

    // uint8 -> float (¿PORQUE?)
    cv::Mat gray(image.rows, image.cols, CV_32FC1);
    gray_UINT8.convertTo(gray, CV_32F);

    std::vector<float> gaussianKernel ({
                                       0.045f, 0.18f, 0.045f,
                                       0.18f, 0.10f, 0.18f,
                                       0.045f, 0.18f, 0.045f
                                      });

    //Output
    cv::Mat harrisCorners(gray.rows, gray.cols, CV_32FC1);


    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    harrisCornersFilter(gray.ptr<float>(),
                        gray.cols,
                        gray.rows,
                        &(gaussianKernel[0]),
                        harrisCorners.ptr<float>());

    Useconds executionTime = std::chrono::duration_cast<Useconds>(
                         std::chrono::system_clock::now() - start);
    std::cout<< "Duracion en GPU: " << (executionTime.count() / 1000) << " ms" << std::endl;


    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display Image", image);

    //cv::namedWindow("Gray Image", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Gray Image", gray);

    //cv::namedWindow("Harris corners Image", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Harris corners Image", harrisCorners);
    //print_features(harrisCorners, gray);
    cv::imwrite("features.tiff", image);


    cv::waitKey(0);

    return 0;
}
