cmake_minimum_required(VERSION 2.8.3)

project(learn_cv)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-old-style-cast -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-recover=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize-recover=address -fno-omit-frame-pointer")

find_package(Boost COMPONENTS filesystem  REQUIRED)
find_package(Eigen REQUIRED)
find_package(CSparse REQUIRED)
find_package(Cholmod REQUIRED)
#find_package(Sophus REQUIRED)

set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
find_package(OpenCV 3 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "        OpenCV Version: ${OpenCV_VERSION}")
    message(STATUS "        OpenCV Include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "        OpenCV Library: ${OpenCV_LIBS}")
endif(OpenCV_FOUND)

set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
find_package(G2O REQUIRED)
file(GLOB G2O_LIBS /home/vance/opt/g2o-20160424/install/lib/*.so)
list(APPEND G2O_LIBS
    cxsparse cholmod
)
if(G2O_FOUND)
    message(STATUS "        G2O Root:    ${G2O_ROOT}")
    message(STATUS "        G2O Include: ${G2O_INCLUDE_DIR}")
    message(STATUS "        G2O Library: ${G2O_LIBS}")
endif(G2O_FOUND)


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}

    ${CSPARSE_INCLUDE_DIR}
    ${Cholmod_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
#    ${Sophus_INCLUDE_DIRS}
)



add_subdirectory(src)
