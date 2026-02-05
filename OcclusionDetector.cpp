#include "OcclusionDetector.h"
#include <stdexcept>
#include <cmath>
#include <thread>

OcclusionDetector::OcclusionDetector(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OcclusionDetector"),
      session_(nullptr),
      session_options_() {

    // session_options_.SetIntraOpNumThreads(1);
    // session_options_.SetGraphOptimizationLevel(
    //     GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    int num_threads = std::thread::hardware_concurrency();
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.AppendExecutionProvider("XNNPACK");

    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    // Input name
    auto input_names = session_.GetInputNames();
    if (input_names.empty()) {
        throw std::runtime_error("ONNX model has no inputs");
    }
    input_name_ = input_names[0];

    // Output names
    auto output_names = session_.GetOutputNames();
    for (const auto& n : output_names) {
        output_names_.push_back(n);
    }
}

std::pair<bool, float> OcclusionDetector::is_occluded(const cv::Mat& img) {
    auto input_data = preprocess(img);

    std::vector<int64_t> input_shape = {1, 3, INPUT_H, INPUT_W};

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name_.c_str()};
    std::vector<const char*> output_names;
    for (const auto& n : output_names_) {
        output_names.push_back(n.c_str());
    }

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names.data(),
        output_names.size()
    );

    float logit = outputs[0].GetTensorMutableData<float>()[0];
    float confidence = sigmoid(logit);

    return {confidence > 0.5f, confidence};
}

std::vector<float> OcclusionDetector::preprocess(const cv::Mat& img) {
    if (img.empty()) {
        throw std::runtime_error("Empty image passed to OcclusionDetector");
    }

    cv::Mat resized_u8;
    cv::resize(
        img,
        resized_u8,
        cv::Size(INPUT_W, INPUT_H),
        0,
        0,
        cv::INTER_LINEAR
    );

    // FORCE uint8 (PIL behavior)
    if (resized_u8.type() != CV_8UC3) {
        resized_u8.convertTo(resized_u8, CV_8UC3);
    }

    // Convert to RGB
    cv::Mat rgb = resized_u8.clone();
    if (resized_u8.channels() == 3) {
        // cv::cvtColor(resized_u8, rgb, cv::COLOR_BGR2RGB);
    } else {
        throw std::runtime_error("Expected 3-channel image");
    }

    // Convert to float [0,1]
    cv::Mat rgb_f;
    if (rgb.type() == CV_8UC3) {
        rgb.convertTo(rgb_f, CV_32FC3, 1.0f / 255.0f);
    } else if (rgb.type() == CV_32FC3) {
        rgb_f = rgb;
    } else {
        throw std::runtime_error("Unsupported image type");
    }

    CV_Assert(rgb_f.isContinuous());

    // CHW
    std::vector<float> input(3 * INPUT_H * INPUT_W);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                input[c * INPUT_H * INPUT_W + h * INPUT_W + w] =
                    rgb_f.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return input;
}

float OcclusionDetector::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
