#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>

typedef unsigned char Byte;

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

    cv::Mat gaussianBlur(gray.rows, gray.cols, CV_8UC1);
    Byte* gPointer = gaussianBlur.ptr<Byte>();

    for (unsigned int i = 1U; i < (gray.rows - 1); ++i)
        for (unsigned int j = 1U; j < (gray.cols - 1); ++j)
        {
            Byte* p = gray.ptr<Byte>(i,j);
            // Middle
            float aux = gaussianKernel[4] * (*p);
            // Stencil
            p = gray.ptr<Byte>(i,j+1);
            aux += gaussianKernel[5] * (*p);
            p = gray.ptr<Byte>(i,j-1);
            aux += gaussianKernel[3] * (*p);
            p = gray.ptr<Byte>(i+1,j);
            aux += gaussianKernel[7] * (*p);
            p = gray.ptr<Byte>(i-1,j);
            aux += gaussianKernel[1] * (*p);
            // Corners
            p = gray.ptr<Byte>(i+1,j+1);
            aux += gaussianKernel[8] * (*p);
            p = gray.ptr<Byte>(i+1,j-1);
            aux += gaussianKernel[6] * (*p);
            p = gray.ptr<Byte>(i-1,j+1);
            aux += gaussianKernel[2] * (*p);
            p = gray.ptr<Byte>(i-1,j-1);
            aux += gaussianKernel[0] * (*p);

            gPointer[i * gray.cols + j] = aux; 
        }

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);

    cv::namedWindow("Gray Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Gray Image", gray);

    cv::namedWindow("Gaussian Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Gaussian Image", gaussianBlur);


    cv::waitKey(0);

    return 0;
}
