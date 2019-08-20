cmake_minimum_required(VERSION 2.8.3)

project(learnCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-old-style-cast -Wno-deprecated-declarations")

set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
find_package(OpenCV 3 REQUIRED)
find_package(Boost COMPONENTS filesystem  REQUIRED)
#find_package(Eigen3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)


add_subdirectory(src)
