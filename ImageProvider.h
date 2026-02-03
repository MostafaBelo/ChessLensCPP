#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

enum class CameraType {
    PI,
    PI_FISH,
    CV2,
    FILES
};

class ImageProvider {
public:
    ImageProvider(CameraType camera,
                  double interval = 0.2,
                  const std::string& data_dir = "");

    // Returns empty Mat if no more images (FILES mode)
    cv::Mat take_image();

    void quit();

private:
    CameraType camera_;
    double interval_;
    std::chrono::steady_clock::time_point last_img_time_;

    // cv2 camera
    cv::VideoCapture cap_;

    // files mode
    std::vector<std::string> imgs_to_load_;

    // optional postprocess
    std::function<cv::Mat(const cv::Mat&)> postprocess_;

    static std::vector<std::string> load_images(const std::string& dir);
};
