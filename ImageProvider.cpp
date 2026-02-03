#include "ImageProvider.h"
#include <filesystem>
#include <thread>
#include <regex>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

static std::vector<std::string> natural_sort(const std::vector<std::string>& files) {
    auto key = [](const std::string& s) {
        std::vector<std::string> parts;
        std::regex re("(\\d+)|(\\D+)");
        auto it = std::sregex_iterator(s.begin(), s.end(), re);
        auto end = std::sregex_iterator();
        for (; it != end; ++it)
            parts.push_back(it->str());
        return parts;
    };

    auto sorted = files;
    std::sort(sorted.begin(), sorted.end(),
        [&](const std::string& a, const std::string& b) {
            auto ka = key(a);
            auto kb = key(b);
            return ka < kb;
        });
    return sorted;
}

std::vector<std::string> ImageProvider::load_images(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& p : fs::directory_iterator(dir)) {
        auto ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            files.push_back(p.path().string());
    }
    return natural_sort(files);
}

ImageProvider::ImageProvider(CameraType camera,
                             double interval,
                             const std::string& data_dir)
    : camera_(camera),
      interval_(interval),
      last_img_time_(std::chrono::steady_clock::time_point::min())
{
    if (camera_ == CameraType::CV2) {
        cap_.open(0);
        if (!cap_.isOpened())
            throw std::runtime_error("Failed to open camera");
    }

    else if (camera_ == CameraType::FILES) {
        if (data_dir.empty() || !fs::is_directory(data_dir))
            throw std::runtime_error("Invalid image directory");
        imgs_to_load_ = load_images(data_dir);
    }

    else if (camera_ == CameraType::PI_FISH) {
        // NOTE:
        // Picamera2 has no official C++ API.
        // You must either:
        // 1. Use libcamera directly (recommended)
        // 2. Wrap Python picamera via IPC
        // 3. Replace with OpenCV + v4l2

        // Fisheye undistortion example (same math as Python)
        cv::FileStorage fs("fisheye_calibration.yaml", cv::FileStorage::READ);
        cv::Mat K, D;
        cv::Size img_size;
        fs["K"] >> K;
        fs["D"] >> D;
        fs["img_size"] >> img_size;

        cv::Mat newK;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            K, D, img_size, cv::Mat::eye(3, 3, CV_64F), newK, 1.0);

        cv::Mat map1, map2;
        cv::fisheye::initUndistortRectifyMap(
            K, D, cv::Mat::eye(3, 3, CV_64F),
            newK, img_size, CV_32FC1, map1, map2);

        postprocess_ = [map1, map2](const cv::Mat& img) {
            cv::Mat out;
            cv::remap(img, out, map1, map2,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            return out;
        };
    }
}

cv::Mat ImageProvider::take_image() {
    if (last_img_time_ != std::chrono::steady_clock::time_point::min()) {
        auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_img_time_).count();
        if (interval_ > 0 && elapsed < interval_) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(interval_ - elapsed));
        }
    }

    cv::Mat img;

    if (camera_ == CameraType::CV2) {
        cap_ >> img;
        if (img.empty())
            throw std::runtime_error("Failed to capture image");
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    else if (camera_ == CameraType::FILES) {
        if (imgs_to_load_.empty())
            return cv::Mat();
        img = cv::imread(imgs_to_load_.front());
        imgs_to_load_.erase(imgs_to_load_.begin());
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    if (postprocess_)
        img = postprocess_(img);

    last_img_time_ = std::chrono::steady_clock::now();
    return img;
}

void ImageProvider::quit() {
    if (camera_ == CameraType::CV2) {
        cap_.release();
        cv::destroyAllWindows();
    }
}
