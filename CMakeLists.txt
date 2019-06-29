cmake_minimum_required(VERSION 2.8.12)
project(sophuspy)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if (APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

find_package(pybind11 REQUIRED)

message("-- cmake python version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.${PYTHON_VERSION_PATCH}")
include_directories(
    ${PROJECT_SOURCE_DIR}/eigen3/
    ${PROJECT_SOURCE_DIR}/sophus/
    ${PROJECT_SOURCE_DIR}/sophus/include/original
    ${PROJECT_SOURCE_DIR}/sophus/include/extension
)

pybind11_add_module(sophuspy sophus/sophuspy.cpp)
