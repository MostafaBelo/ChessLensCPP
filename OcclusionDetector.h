#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

/**
 * Occlusion detector using ONNX Runtime
 * Matches Python OcclusionDetectorCNN (ONNX path)
 */
class OcclusionDetector {
public:
    explicit OcclusionDetector(const std::string& model_path);

    /**
     * @param img Input image (BGR or RGB, CV_8UC3 or CV_32FC3)
     * @return {is_occluded, confidence}
     */
    std::pair<bool, float> is_occluded(const cv::Mat& img);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::SessionOptions session_options_;

    std::string input_name_;
    std::vector<std::string> output_names_;

    // Produces [1, 3, 240, 240] float tensor data
    std::vector<float> preprocess(const cv::Mat& img);

    static float sigmoid(float x);

    static constexpr int INPUT_W = 240;
    static constexpr int INPUT_H = 240;
};
