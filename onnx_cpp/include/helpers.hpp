#include <vector>
#include <opencv2/opencv.hpp>

std::vector<float> matToVector(const cv::Mat& img);

cv::Mat tensorToMat(const float* tensor_data, int width, int height);