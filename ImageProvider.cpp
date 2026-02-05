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

void ImageProvider::setup_pi_camera() {
    // piCam_ = std::make_unique<LibCameraCapture>(1280, 720);
    piCam_ = std::make_unique<LibCameraCapture>(1920, 1080);
    // piCam_ = std::make_unique<LibCameraCapture>(2592, 1944);
    // piCam_ = std::make_unique<LibCameraCapture>(1640, 1232);

    // // Raspberry Pi cameras via libcamera/V4L2 
    // // Usually index 0, but we force V4L2 backend
    // cap_.open(0, cv::CAP_V4L2);
    
    // if (!cap_.isOpened()) {
    //     throw std::runtime_error("Failed to open Pi Camera via V4L2. Ensure libcamera is not hogging the device.");
    // }

    // // Set resolution (adjust to your Pi camera's native aspect ratio)
    // cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    // cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    // cap_.set(cv::CAP_PROP_FPS, 30);
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

    else if (camera_ == CameraType::PI || camera_ == CameraType::PI_FISH) {
        setup_pi_camera();

        if (camera_ == CameraType::PI_FISH) {
            // Load fisheye calibration
            cv::FileStorage storage("models/fisheye_calibration.yaml", cv::FileStorage::READ);
            if(storage.isOpened()) {
                cv::Mat K, D;
                cv::Size img_size;
                storage["K"] >> K;
                storage["D"] >> D;
                storage["img_size"] >> img_size;

                cv::Mat newK, map1, map2;
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, img_size, cv::Mat::eye(3, 3, CV_64F), newK, 1.0);
                cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_64F), newK, img_size, CV_32FC1, map1, map2);

                postprocess_ = [map1, map2](const cv::Mat& img) {
                    cv::Mat out;
                    cv::remap(img, out, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
                    return out;
                };
            }
        }
    }

    else if (camera_ == CameraType::FILES) {
        if (data_dir.empty() || !fs::is_directory(data_dir))
            throw std::runtime_error("Invalid image directory");
        imgs_to_load_ = load_images(data_dir);
    }
}

cv::Mat ImageProvider::take_image() {
    auto now = std::chrono::steady_clock::now();
    if (last_img_time_ != std::chrono::steady_clock::time_point::min()) {
        auto elapsed = std::chrono::duration<double>(
            now - last_img_time_).count();
        if (interval_ > 0 && elapsed < interval_) {
            auto sleep_period = interval_ - elapsed;
            total_wait_time += sleep_period;
            std::this_thread::sleep_for(
                std::chrono::duration<double>(sleep_period));
        }
    }
    last_img_time_ = now;

    cv::Mat img;

    if (camera_ == CameraType::CV2) {
        cap_ >> img;
        if (img.empty())
            throw std::runtime_error("Failed to capture image");
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    else if (camera_ == CameraType::PI || camera_ == CameraType::PI_FISH) {
        img = piCam_->capture();
        if (img.empty())
            throw std::runtime_error("Failed to capture image");
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
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

    return img;
}

void ImageProvider::quit() {
    // if (camera_ == CameraType::CV2) {
    //     cap_.release();
    //     cv::destroyAllWindows();
    // }
    if (piCam_) {
        piCam_.reset();
    }
    if (cap_.isOpened()) {
        cap_.release();
    }
}
