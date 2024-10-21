FROM pytorch/pytorch:latest

# Set non-interactive frontend for apt-get to avoid issues with tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libblas-dev \
    libatlas-base-dev \
    libopencv-dev \
    python3 \
    python3-pip \
    wget \
    unzip

# Set the working directory
WORKDIR /app

# Copy the current directory into the container
COPY . /app

# Set environment variable for the serialized model file
ENV SERIALIZED_MODEL_FILE=traced_resnet_model.pt

# Set the CMAKE_PREFIX_PATH environment variable to point to LibTorch
ENV CMAKE_PREFIX_PATH=/usr/local/libtorch

# Download and extract the LibTorch library (CPU version)
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip libtorch-shared-with-deps-latest.zip && \
    rm libtorch-shared-with-deps-latest.zip && \
    mv libtorch /usr/local/libtorch

# Make sure the shared libraries are loaded
RUN ldconfig

# Make the compile_and_run.sh script executable
RUN chmod +x compile_and_run.sh

# Run the compile_and_run.sh script to build the application
RUN ./compile_and_run.sh

# Default command to keep the container running or enter the bash shell
CMD ["bash"]
