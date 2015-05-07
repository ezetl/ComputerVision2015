#include <stdio.h>
#include <vector>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, char** argv )
{
  if (argc != 3) {
    std::cout << "usage: Homography <Image_1> <Image_2>" << std::endl;
    return -1;
  }

  // lee imágenes del disco y convierte a escala de grises
  cv::Mat im1_rgb, im1;
  im1_rgb = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (im1_rgb.data == NULL) {
    std::cout << "error reading " << argv[1] << std::endl;
    return -1;
  }
  cv::cvtColor(im1_rgb, im1, CV_BGR2GRAY);

  cv::Mat im2_rgb, im2;
  im2_rgb = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
  if (im2_rgb.data == NULL) {
    std::cout << "error reading " << argv[2] << std::endl;
    return -1;
  }
  cv::cvtColor(im2_rgb, im2, CV_BGR2GRAY);

  //---------------------------------
  // SIFT
  //---------------------------------

  // // detect keypoints
  // cv::SiftFeatureDetector detector;
  // std::vector<cv::KeyPoint> kp1, kp2;
  // detector.detect(im1, kp1);
  // detector.detect(im2, kp2);

  // // compute descriptores
  // cv::SiftDescriptorExtractor extractor;
  // cv::Mat desc1, desc2;
  // extractor.compute(im1, kp1, desc1);
  // extractor.compute(im2, kp2, desc2);

  cv::ORB feat;
  std::vector<cv::KeyPoint> kp1, kp2;
  feat.detect(im1, kp1);
  feat.detect(im2, kp2);

  cv::Mat desc1, desc2;
  feat.compute(im1, kp1, desc1);
  feat.compute(im2, kp2, desc2);

  std::cout << kp1.size() << " points @ im1" << std::endl;
  std::cout << kp2.size() << " points @ im2" << std::endl;

  // visualiza keypoints
  cv::Mat im1_sift, im2_sift;
  cv::drawKeypoints(im1, kp1, im1_sift, cv::Scalar(0,255,0), 4);
  cv::drawKeypoints(im2, kp2, im2_sift, cv::Scalar(0,255,0), 4);

  cv::namedWindow("SIFT KeyPoints @ im1", cv::WINDOW_AUTOSIZE);
  cv::imshow("SIFT KeyPoints @ im1", im1_sift);

  cv::namedWindow("SIFT KeyPoints @ im2", cv::WINDOW_AUTOSIZE);
  cv::imshow("SIFT KeyPoints @ im2", im2_sift);

  //---------------------------------
  // Matching
  //---------------------------------

  //cv::BFMatcher matcher(cv::NORM_L2);
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<cv::DMatch> matches;
  matcher.match(desc1, desc2, matches);

  // EJERCICIO 1: implementar matching por umbral sobre la relación entre
  // distancias al primero y segundo mejor descriptor.
  // 1- calcular la distancia de un descriptor contra todos los de la otra imagen
  // 2- me quedo con la mejor (menor)
  // 3- ademas guardo la segunda mejor distancia
  // 4- creo un numero usando esas dos distancias: 1ra/2da. Si ese numero es menor que el threshold que elegi para filtrar distancias
  //    me quedo con eso? chequear
  // Se puede reutilizar el matcher y reordenarlo


  // visualiza correspondencias
  cv::Mat im_matches;
  cv::drawMatches(im1, kp1, im2, kp2, matches, im_matches);

  std::cout << matches.size() << " matches" << std::endl;
  cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE);
  cv::imshow("Matches", im_matches);

  // -------------------------------
  // Homography
  // -------------------------------

  std::vector<cv::Point2f> points1, points2;
  for (int i = 0; i < matches.size(); i++) {
    points1.push_back(kp1[matches[i].queryIdx].pt);
    points2.push_back(kp2[matches[i].trainIdx].pt);
  }
  cv::Mat H = cv::findHomography(points2, points1, CV_RANSAC, 1);

  // EJERCICIO 2: computar error de reproyección promedio considerando: a) todos
  // los pares de puntos, b) solo los inliers, en base a la homografía estimada.
  // Comparar los métodos de matching.
  // error de reproyeccion: distancia euclidea entre el punto original y el de destino reproyectado en la imagen original (usando la inversa de la matriz H aplicada a x2 -punto de destino-).
  // H.inverse o algo asi para calcular la inversa. 
  // para considerar los inliers, tomar los menores que el threshold 1 (usado en findHomography)

  // warping
  cv::Mat im_warp;
  int warp_width = 1.8 * im1.cols;
  int warp_height = 1.5 * im1.rows;
  cv::warpPerspective(im2_rgb, im_warp, H, cv::Size(warp_width, warp_height));

  // EJERCICIO 3: calcular tamaño óptimo de la imagen transformada para que se vea
  // la imagen transformada *completa*. TIP: transformar el sistema de
  // coordenadas de im1 e im2 al centro de la imagen y aplicar la transformación
  // en ese marco de referencia.

  // crear una nueva H 3x3 que componga HHtrans. 



  // blending
  float alpha = 0.25;
  cv::Mat view = im_warp(cv::Range(0, im1.rows), cv::Range(0, im1.cols));
  view = alpha*im1_rgb + (1.0-alpha)*view;
  cv::namedWindow("Warp", cv::WINDOW_AUTOSIZE);
  cv::imshow("Warp",im_warp);

  // EJERCICIO 4: repetir pipeline usando otros pares de detector/descriptor:
  // SURF, ORB y FAST+FREAK. Cuidado con la métrica de comparación.
  //
  // Referencias:
  //   Bay, H. and Tuytelaars, T. and Van Gool, L. “SURF: Speeded Up Robust Features”. ECCV 2006
  //   Rublee, E. and Rabaud, V. and Konolige, K. and Bradski, G. "ORB: An efficient alternative to SIFT or SURF". ICCV 2011
  //   Rosten, E. and Drummond. T. "Machine learning for high-speed corner detection". ECCV 2006
  //   Alahi, A. and Ortiz, R. and Vandergheynst, P. "FREAK: Fast Retina Keypoint". CVPR 2012

  // espera que se presiones una tecla
  std::cout << "\nPresione cualquier tecla para salir..." << std::endl;
  cv::waitKey(0);
  return 0;
}