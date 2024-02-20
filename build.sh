#!/bin/bash

# mkdir -p build && cd build
# cmake ..
# make -j4

mkdir -p pybind11/build && cd pybind11/build
cmake ..
make -j4
sudo make install
