#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>
#include <omp.h>

/*
g++ -o HarrisCorners  HarrisCorners.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lgomp
 */

#define SIGMA 9.0
#define SIGMA1 2.0
#define K_ 0.04
#define THRESHOLD 100.0

typedef unsigned char Byte;


void gradientSobel(cv::Mat& img, cv::Mat& img_x, cv::Mat& img_y) {
  assert((img.rows == img_x.rows) && (img.cols == img_x.cols) && \
         (img.cols == img_y.cols) && (img.cols == img_y.cols));

  // zero padding
  cv::Mat padding(img.rows, img.cols, CV_32FC1);
  copyMakeBorder(img, padding, 1, 1, 1, 1, IPL_BORDER_CONSTANT, cv::Scalar(0));

  // Se puede usar OpenMP para acelerar un poco las cosas.
  // Descomentar la linea de abajo para usar OpenMP:
  //# pragma omp parallel for shared(padding, img_x, img_y)
  for (int i = 1; i < padding.rows-1; ++i) {
    float* img_0 = padding.ptr<float>(i - 1);
    float* img_1 = padding.ptr<float>(i);
    float* img_2 = padding.ptr<float>(i + 1);

    // le resto la columna agregada por el padding
    float* img_x_1 = img_x.ptr<float>(i-1);
    float* img_y_1 = img_y.ptr<float>(i-1);

    for (int j = 1; j < padding.cols-1; ++j) {
      img_x_1[j-1] = - img_0[j - 1] + img_0[j + 1] - 2.0 * img_1[j - 1] +
          2.0 * img_1[j + 1] - img_2[j - 1] + img_2[j + 1];

      img_y_1[j-1] = - img_0[j - 1] - 2.0 * img_0[j] - img_0[j + 1] +
          img_2[j - 1] + 2.0 * img_2[j] + img_2[j + 1];
    }
  }
}


void gradientSobel_separable(cv::Mat& img, cv::Mat& img_x, cv::Mat& img_y) {
  assert((img.rows == img_x.rows) && (img.cols == img_x.cols) && \
         (img.cols == img_y.cols) && (img.cols == img_y.cols));

  // zero padding
  cv::Mat padding(img.rows, img.cols, CV_32FC1);
  copyMakeBorder(img, padding, 1, 1, 1, 1, IPL_BORDER_CONSTANT, cv::Scalar(0));

  cv::Mat aux_x(padding.rows, padding.cols, CV_32FC1);
  cv::Mat aux_y(padding.rows, padding.cols, CV_32FC1);

  //1er pasada
  //# pragma omp parallel for 
  for (int i = 1; i < padding.rows-1; ++i) {
    float* img_1 = padding.ptr<float>(i);

    float* aux_x_1 = aux_x.ptr<float>(i);
    float* aux_y_1 = aux_y.ptr<float>(i);

    for (int j = 1; j < padding.cols-1; ++j) {
      aux_x_1[j] = - img_1[j - 1] + img_1[j + 1]; 
      aux_y_1[j] = img_1[j - 1] + 2.0 * img_1[j] + img_1[j + 1];
    }
  }

  // 2da pasada
  //# pragma omp parallel for 
  for (int i = 1; i < padding.rows-1; ++i) {
    float* img_0_x = aux_x.ptr<float>(i - 1);
    float* img_1_x = aux_x.ptr<float>(i);
    float* img_2_x = aux_x.ptr<float>(i + 1);
    float* img_0_y = aux_y.ptr<float>(i - 1);
    float* img_2_y = aux_y.ptr<float>(i + 1);

    float* img_x_1 = img_x.ptr<float>(i-1);
    float* img_y_1 = img_y.ptr<float>(i-1);

    for (int j = 1; j < padding.cols-1; ++j) {
      img_x_1[j-1] = img_0_x[j] + 2.0 * img_1_x[j] + img_2_x[j]; 
      img_y_1[j-1] = - img_0_y[j] + img_2_y[j + 1];
    }
  }
}


void local_maxima_3x3(cv::Mat& img, std::vector<cv::Point>& points) {
  points.clear();
  for (int i = 1; i < img.rows - 1; ++i) {
    // punteros a fila anterior, actual y siguiente.
    float* img_0 = img.ptr<float>(i - 1);
    float* img_1 = img.ptr<float>(i);
    float* img_2 = img.ptr<float>(i + 1);
    for (int j = 1; j < img.cols - 1; ++j) {
        float max = img_1[j];
        bool is_max = true;
        for (int l=j-1; l<j+2 && is_max; ++l)
            if(l==j)
                is_max = (max>img_0[l] && max>img_2[l]); 
            else
                is_max = (max>img_0[l] && max>img_1[l] && max>img_2[l]);
        if(is_max)
            points.push_back(cv::Point(j,i));
    }
  }
}


void print_features(cv::Mat& dest, std::vector<cv::Point>& points)
{
  std::vector<cv::Point>::iterator it;
  for(it=points.begin(); it!=points.end(); ++it)
    cv::circle(dest, cv::Point( (*it).x, (*it).y ), 5, cv::Scalar(0), 2, 8, 0);
}


int main(int argc, char** argv )
{
  if (argc != 3) {
    std::cout << "usage: HarrisCorners <Image_Path>" << std::endl;
    return -1;
  }

  // lee imagen del disco
  cv::Mat image;
  image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (image.data == NULL) {
    std::cout << "No image data" << std::endl;
    return -1;
  }

  // convierte a escala de grises (uint8)
  cv::Mat gray;
  cv::cvtColor(image, gray, CV_BGR2GRAY);

  // Descomentar para sacar un poco mas de ruido:
  //GaussianBlur(gray, gray, cv::Size(), SIGMA1, SIGMA1, cv::BORDER_DEFAULT);

  // uint8 -> float
  cv::Mat gray_(image.rows, image.cols, CV_32FC1);
  gray.convertTo(gray_, CV_32F);

  cv::Mat im_x(image.rows, image.cols, CV_32FC1);
  cv::Mat im_y(image.rows, image.cols, CV_32FC1);
  // computo de derivadas espaciales
  gradientSobel_separable(gray_, im_x, im_y);
  //gradientSobel(gray_, im_x, im_y);

  cv::Mat a11 = im_x.mul(im_x);
  cv::Mat a12 = im_x.mul(im_y);
  cv::Mat a22 = im_y.mul(im_y);

  // integracion local = Gaussian blur
  cv::Mat a11w(image.rows, image.cols, CV_32FC1);
  cv::Mat a12w(image.rows, image.cols, CV_32FC1);
  cv::Mat a22w(image.rows, image.cols, CV_32FC1);
  GaussianBlur(a11, a11w, cv::Size(), SIGMA, SIGMA, cv::BORDER_DEFAULT);
  GaussianBlur(a12, a12w, cv::Size(), SIGMA, SIGMA, cv::BORDER_DEFAULT);
  GaussianBlur(a22, a22w, cv::Size(), SIGMA, SIGMA, cv::BORDER_DEFAULT);

  // computo de funcion R
  cv::Mat R = a11w.mul(a22w) - a12w.mul(a12w) - K_ * (a11w + a22w).mul(a11w + a22w);

  // umbralizado
  cv::threshold(R, R, THRESHOLD, 0.0, cv::THRESH_TOZERO);

  // non-maxima supression (NMS)
  std::vector<cv::Point> harris_corners;
  local_maxima_3x3(R, harris_corners);

  // R al rango [0, 1] antes de visualizar
  double min;
  double max;
  cv::minMaxIdx(R, &min, &max);
  cv::Mat R_;
  R.convertTo(R_, CV_32FC1, 1.0 / (max - min), -min / (max - min));
  cv::minMaxIdx(R_, &min, &max);

  // Visualizacion
  cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Image", image);

  cv::namedWindow("Cornerness", cv::WINDOW_AUTOSIZE);
  cv::imshow("Cornerness", R_);

  cv::Mat features = image;
  print_features(features, harris_corners);
  cv::namedWindow("Features", cv::WINDOW_AUTOSIZE);
  cv::imshow("Features", features);

  cv::waitKey(0);

  return 0;
}
