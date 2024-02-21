#!/bin/bash

mkdir -p pybind11/build && cd pybind11/build
cmake -DBUILD_TESTING=OFF ..
make -j4
sudo make install
