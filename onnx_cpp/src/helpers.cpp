#include <helpers.hpp>

// Helper function to convert OpenCV Mat to a flat vector<float>
std::vector<float> matToVector(const cv::Mat& img) {
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 255.0); // Normalize to [0,1]
    return std::vector<float>(img_float.begin<float>(), img_float.end<float>());
}

// Helper function to convert ONNX output tensor to cv::Mat
cv::Mat tensorToMat(const float* tensor_data, int width, int height) {
    cv::Mat img_out_y(height, width, CV_32FC1);
    memcpy(img_out_y.data, tensor_data, width * height * sizeof(float));
    img_out_y.convertTo(img_out_y, CV_8U, 255.0); // Scale back to [0, 255]
    return img_out_y;
}