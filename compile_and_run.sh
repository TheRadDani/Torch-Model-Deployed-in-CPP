#!/bin/bash
mkdir -p build
cd ./build
export CMAKE_PREFIX_PATH=~/cuda-cpp/libtorch
cmake ..
cmake --build . --config Release
