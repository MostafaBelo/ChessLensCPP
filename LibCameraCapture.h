#pragma once

#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>

class LibCameraCapture {
public:
    LibCameraCapture(int width = 1280, int height = 720);
    ~LibCameraCapture();

    cv::Mat capture();

private:
    void requestComplete(libcamera::Request *request);

    std::unique_ptr<libcamera::CameraManager> cameraManager_;
    std::shared_ptr<libcamera::Camera> camera_;
    libcamera::Stream *stream_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;

    std::vector<std::unique_ptr<libcamera::Request>> requests_;

    cv::Mat lastFrame_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool frameReady_ = false;

    int targetWidth_;
    int targetHeight_;

    std::atomic<bool> running_{true};
};
