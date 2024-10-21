#!/bin/bash

# Set the mode to "annotation" by default
mode="tracing"

# If an argument is passed, use it as the mode (either 'tracing' or 'annotation')
if [ $# -gt 0 ]; then
    if [ "$1" == "tracing" ] || [ "$1" == "annotation" ]; then
        mode=$1
    else
        echo "Invalid argument. Use 'tracing' or 'annotation'."
        exit 1
    fi
fi

# Run the appropriate Python script based on the mode
if [ "$mode" == "tracing" ]; then
    echo "Running TorchScript in tracing mode..."
    python3 TorchScriptTracing.py
else
    echo "Running TorchScript in annotation mode..."
    python3 TorchScriptAnnotation.py
fi

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd ./build

# Set CMAKE_PREFIX_PATH for libtorch

if [ -z "$CMAKE_PREFIX_PATH" ]; then
    export CMAKE_PREFIX_PATH=~/cuda-cpp/libtorch
fi

# Run CMake commands
cmake ..
cmake --build . --config Release

# Check if SERIALIZED_MODEL is set, if not, assign a default value
if [ -z "$SERIALIZED_MODEL_FILE" ]; then
    # Default value for the serialized model file
    SERIALIZED_MODEL_FILE="traced_resnet_model.pt"
fi

# Copy the app and run it with the serialized model
cp ./app ..

cd ..

./app "$SERIALIZED_MODEL_FILE"
