cmake_minimum_required(VERSION 2.8)
project( HarrisCorners )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_definitions(-std=c++0x)

add_executable( HarrisCorners HarrisCorners.cpp )
add_executable( ch ch.cpp )
target_link_libraries( HarrisCorners ${OpenCV_LIBS} )
target_link_libraries( ch ${OpenCV_LIBS} )

