#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <chrono>

#include <helpers.hpp>

using namespace cv;
using namespace std;

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXSuperResolution");

    // Load ONNX model
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "../../super_resolution.onnx", session_options);

    // Load the image using OpenCV and preprocess
    cv::Mat img = cv::imread("../../images/cat.jpg");
    if (img.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }

    // Resize the image to model's input size (224x224) and convert to YCbCr
    cv::resize(img, img, cv::Size(224, 224));
    cv::Mat img_ycbcr, img_y;
    cv::cvtColor(img, img_ycbcr, COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(img_ycbcr, channels);
    img_y = channels[0]; // Y channel

    // Convert the Y channel to a flat vector<float>
    std::vector<float> input_data = matToVector(img_y);

    // Prepare the input tensor
    std::vector<int64_t> input_shape = {1, 1, img_y.rows, img_y.cols};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                             OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(),
                                                            input_data.size(), 
                                                            input_shape.data(),
                                                            input_shape.size());

    // Retrieve input and output names
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    // Inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                                    input_names, &input_tensor,
                                                    1,
                                                    output_names,
                                                    1);

    // Extract and post-process the output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    int output_width = img_y.cols * 2; // Adjust width based on super-resolution scale
    int output_height = img_y.rows * 2; // Adjust height based on super-resolution scale

    // Convert tensor output back to an image
    cv::Mat img_out_y = tensorToMat(output_data, output_width, output_height);

    // Resize Cb and Cr channels to match output size and merge channels
    cv::resize(channels[1], channels[1], cv::Size(output_width, output_height), 0, 0, INTER_CUBIC);
    cv::resize(channels[2], channels[2], cv::Size(output_width, output_height), 0, 0, INTER_CUBIC);

    cv::Mat img_output;
    cv::merge(std::vector<cv::Mat>{img_out_y, channels[1], channels[2]}, img_output);
    cv::cvtColor(img_output, img_output, COLOR_YCrCb2BGR);

    // Save the result
    cv::imwrite("../../images/cat_superres_with_ort.jpg", img_output);
    std::cout << "Inference completed and result saved!" << std::endl;

    // Measure inference time
    auto start = chrono::high_resolution_clock::now();
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    std::cout << "ONNX model inference time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
