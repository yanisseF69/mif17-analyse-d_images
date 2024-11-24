cmake_minimum_required(VERSION 3.0.0)
project( TP1 )


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( TP1 TP1.cpp )
target_link_libraries( TP1 ${OpenCV_LIBS} )