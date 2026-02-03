#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include "LibCameraCapture.h"

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
    
    cv::Mat take_image();
    void quit();

private:
    CameraType camera_;
    double interval_;
    std::chrono::steady_clock::time_point last_img_time_;
    
    std::unique_ptr<LibCameraCapture> piCam_;
    cv::VideoCapture cap_;
    std::vector<std::string> imgs_to_load_;
    std::function<cv::Mat(const cv::Mat&)> postprocess_;

    static std::vector<std::string> load_images(const std::string& dir);
    void setup_pi_camera();
};