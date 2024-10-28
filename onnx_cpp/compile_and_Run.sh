# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd ./build

cmake ..

make

# Return to project folder
./onnx_cpp