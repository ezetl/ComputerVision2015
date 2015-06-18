#include <stdio.h>
#include <vector>
#include <cassert>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>


//EJERCICIO 2
float euclidean_distance(float x1, float y1, float x2, float y2)
{
  return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

cv::Point3f multiply_byH(cv::Mat H, cv::Point2f p){
  return cv::Point3f(H.at<double>(0,0)*p.x + H.at<double>(0,1)*p.y + H.at<double>(0,2),
                     H.at<double>(1,0)*p.x + H.at<double>(1,1)*p.y + H.at<double>(1,2),
                     H.at<double>(2,0)*p.x + H.at<double>(2,1)*p.y + H.at<double>(2,2));
}

float reprojection_error(cv::Mat& H, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2)
{
  cv::Mat H_inv = H.inv();
  float mean = 0.0;
  for(int i=0; i<points2.size(); ++i)
  {
    // Reproyeccion. coord. homogenea
    cv::Point3f r = multiply_byH(H.inv(), points2[i]);
    // Tengo que dividir por la tercera coordenada para des-homogeneizar
    mean += euclidean_distance(r.x/r.z, r.y/r.z, points1[i].x, points1[i].y);
  }
  std::cout<<"mean: "<<mean<<" points2 size: "<<points2.size()<<std::endl;
  return mean / (float) points2.size();
}


float reprojection_error2(cv::Mat& H, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2)
{
  double mean = 0.0;
  for(int i=0; i<points2.size(); ++i)
  {
    // Reproyeccion. coord. homogenea
    cv::Point3f r = multiply_byH(H, points1[i]);
    std::cout<<"Hpoints1 x: "<<r.x<<" Hpoints1 y: "<<r.y<<" Hpoints1 z: "<<r.z<<" points2 x: "<<points2[i].x<<" points2 y: "<<points2[i].y<<std::endl;
    // Tengo que dividir por la tercera coordenada para des-homogeneizar
    mean += euclidean_distance(r.x/r.z, r.y/r.z, points2[i].x, points2[i].y);
  }
  return mean / (float) points2.size();
}

float reprojection_error_inliers(cv::Mat& H,
                             std::vector<cv::Point2f> points1,
                             std::vector<cv::Point2f> points2,
                             const float threshold)
{
  //inliers: solo sumo los terminos tales que ||x1 - H.inv()*x2||<1
  cv::Mat H_inv = H.inv();
  float mean = 0.0;
  unsigned int N = 0;
  for(int i=0; i<points2.size(); ++i)
  {
    // primero ver si es un inlier
    cv::Point3f r0 = multiply_byH(H, points1[i]);
    //distancia con respecto a el points2[i] correspondiente
    float dist = euclidean_distance(r0.x/r0.z, r0.y, points2[i].x, points2[i].y);
    if(dist<=threshold){
      // Reproyeccion. coord. homogenea
      cv::Point3f r = multiply_byH(H.inv(), points2[i]);
      // Tengo que dividir por la tercera coordenada para des-homogeneizar la coordenada
      mean += euclidean_distance(r.x/r.z, r.y/r.z, points1[i].x, points1[i].y);
      N++;
    }
  }
  if(N==0){
    std::cout<<"N igual a cero."<<std::endl;
  }
  return mean / (float) N;
}

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
  // SURF 
  //---------------------------------

  // detect keypoints
  cv::SurfFeatureDetector detector;
  std::vector<cv::KeyPoint> kp1, kp2;
  detector.detect(im1, kp1);
  detector.detect(im2, kp2);

  // compute descriptores
  cv::SurfDescriptorExtractor extractor;
  cv::Mat desc1, desc2;
  extractor.compute(im1, kp1, desc1);
  extractor.compute(im2, kp2, desc2);

  std::cout << kp1.size() << " points @ im1" << std::endl;
  std::cout << kp2.size() << " points @ im2" << std::endl;

  // visualiza keypoints
  cv::Mat im1_sift, im2_sift;
  cv::drawKeypoints(im1, kp1, im1_sift, cv::Scalar(0,255,0), 4);
  cv::drawKeypoints(im2, kp2, im2_sift, cv::Scalar(0,255,0), 4);

  cv::namedWindow("SURF KeyPoints @ im1", cv::WINDOW_AUTOSIZE);
  cv::imshow("SURF KeyPoints @ im1", im1_sift);

  cv::namedWindow("SURF KeyPoints @ im2", cv::WINDOW_AUTOSIZE);
  cv::imshow("SURF KeyPoints @ im2", im2_sift);

  //---------------------------------
  // Matching
  //---------------------------------
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  matcher.match(desc1, desc2, matches);

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
    //queryIdx es el de origen
    points1.push_back(kp1[matches[i].queryIdx].pt);
    // trainIdx es el de destino
    points2.push_back(kp2[matches[i].trainIdx].pt);
  }
  assert(points1.size() == points2.size());
  cv::Mat H = cv::findHomography(points2, points1, CV_RANSAC, 1);

  // EJERCICIO 2
  float error = reprojection_error(H, points1, points2);
  //float error = reprojection_error_inliers(H, points1, points2, 1.0);
  std::cout<<"Error de reproyeccion: "<<error<<std::endl;

  // warping
  cv::Mat im_warp;

  // EJERCICIO 3
  //Veo a donde manda H los puntos de las esq.
  cv::Point2f p1(0.0,0.0);
  cv::Point2f p2((float)im1.cols, 0.0);
  cv::Point2f p3((float)im1.cols, (float)im1.rows);
  cv::Point2f p4(0.0, (float)im1.rows);
  cv::Point3f pf1 = multiply_byH(H,p1);
  cv::Point3f pf2 = multiply_byH(H,p2);
  cv::Point3f pf3 = multiply_byH(H,p3);
  cv::Point3f pf4 = multiply_byH(H,p4);

  //calculo el alto y ancho
  float minx,miny,maxx,maxy;
  minx = std::min(pf1.x/pf1.z, pf2.x/pf2.z); minx = std::min(minx, pf3.x/pf3.z); minx = std::min(minx, pf4.x/pf4.z);
  maxx = std::max(pf1.x/pf1.z, pf2.x/pf2.z); maxx = std::max(maxx, pf3.x/pf3.z); maxx = std::max(maxx, pf4.x/pf4.z);
  miny = std::min(pf1.y/pf1.z, pf2.y/pf2.z); miny = std::min(miny, pf3.y/pf3.z); miny = std::min(miny, pf4.y/pf4.z);
  maxy = std::max(pf1.y/pf1.z, pf2.y/pf2.z); maxy = std::max(maxy, pf3.y/pf3.z); maxy = std::max(maxy, pf4.y/pf4.z);

  int warp_width = 1.8 * im1.cols;
  int warp_height = 1.8 * im1.rows;

  //armo la H de translacion
  cv::Mat Ht(3,3, CV_64FC1); 
  Ht.at<double>(0,0) = 1.0;
  Ht.at<double>(0,1) = 0.0;
  Ht.at<double>(0,2) = -minx;
  Ht.at<double>(1,0) = 0.0;
  Ht.at<double>(1,1) = 1.0;
  Ht.at<double>(1,2) = -miny;
  Ht.at<double>(2,0) = 0.0;
  Ht.at<double>(2,1) = 0.0;
  Ht.at<double>(2,2) = 1.0;

  cv::warpPerspective(im2_rgb, im_warp, Ht*H, cv::Size(warp_width, warp_height));

  // blending
  float alpha = 0.25;
  cv::Mat im_warp2;
  cv::warpPerspective(im1_rgb, im_warp2, Ht, cv::Size(warp_width, warp_height));
  cv::Mat view = im_warp(cv::Range(0, im1.rows), cv::Range(0, im1.cols));
  cv::Mat view2 = im_warp2(cv::Range(0, im2.rows), cv::Range(0, im2.cols));
  view = alpha*view2 + (1.0-alpha)*view;
  cv::namedWindow("Warp", cv::WINDOW_AUTOSIZE);
  cv::imshow("Warp",im_warp);

  std::cout << "\nPresione cualquier tecla para salir..." << std::endl;
  cv::waitKey(0);
  return 0;
}
