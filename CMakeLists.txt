cmake_minimum_required(VERSION 2.8.12)
project(sophus)

set(CMAKE_BUILD_TYPE Release)
set(PYBIND11_CPP_STANDARD -std=c++11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include/original
    ${PROJECT_SOURCE_DIR}/include/extension
)

pybind11_add_module(sophus sophuspy.cpp)