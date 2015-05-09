cmake_minimum_required(VERSION 2.8)
project( Homography )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_definitions(-std=c++0x)

add_executable( Homography Homography.cpp )
target_link_libraries( Homography ${OpenCV_LIBS} )

