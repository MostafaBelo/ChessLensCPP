#include "LibCameraCapture.h"
#include <iostream>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <thread>
#include <chrono>

using namespace libcamera;

LibCameraCapture::LibCameraCapture(int width, int height): targetWidth_(width), targetHeight_(height) {
    cameraManager_ = std::make_unique<CameraManager>();
    cameraManager_->start();

    if (cameraManager_->cameras().empty())
        throw std::runtime_error("No cameras found");

    camera_ = cameraManager_->cameras()[0];
    camera_->acquire();

    std::unique_ptr<CameraConfiguration> config =
        camera_->generateConfiguration({ StreamRole::StillCapture });

    StreamConfiguration &streamConfig = config->at(0);
    // streamConfig.size.width = width;
    // streamConfig.size.height = height;
    // streamConfig.pixelFormat = formats::YUV420;
    streamConfig.pixelFormat = formats::RGB888;
    streamConfig.bufferCount = 4;

    CameraConfiguration::Status status = config->validate();
    std::cout << "Status: " << (status == CameraConfiguration::Valid ? "Valid" : 
                                status == CameraConfiguration::Adjusted ? "Adjusted" : "Invalid") << std::endl;
    std::cout << "Final: " << streamConfig.toString() << std::endl;
    // std::cout << "Stride: " << streamConfig.stride << std::endl;
    if (status == CameraConfiguration::Invalid) {
        throw std::runtime_error("Invalid camera configuration");
    }

    camera_->configure(config.get());
    stream_ = streamConfig.stream();

    allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
    allocator_->allocate(stream_);

    for (const auto &buffer : allocator_->buffers(stream_)) {
        std::unique_ptr<Request> request = camera_->createRequest();

        // Enable AE + AWB
        request->controls().set(controls::AeEnable, true);
        request->controls().set(controls::AwbEnable, true);
        
        // CRITICAL: Set crop to full sensor on EVERY request
        // if (pixelArrayOpt) {
        //     request->controls().set(controls::ScalerCrop, fullSensorCrop);
        // }

        request->addBuffer(stream_, buffer.get());
        requests_.push_back(std::move(request));
    }

    camera_->requestCompleted.connect(this, &LibCameraCapture::requestComplete);


    camera_->start();

    for (int i = 0; i < 20; ++i) {
        camera_->queueRequest(requests_[i % requests_.size()].get());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    for (auto &req : requests_)
        camera_->queueRequest(req.get());

    // Wait for AE to converge: discard first 10–15 frames
    for (int i = 0; i < 10; ++i)
        capture();

    std::cout << "=== Camera initialized ===" << std::endl;
}

LibCameraCapture::~LibCameraCapture() {
    running_ = false;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    camera_->stop();
    camera_->release();
    cameraManager_->stop();
}

void LibCameraCapture::requestComplete(Request *request) {
    if (request->status() != Request::RequestComplete)
        return;

    const FrameBuffer *buffer = request->buffers().begin()->second;
    // const FrameMetadata &metadata = buffer->metadata();
    const libcamera::StreamConfiguration &cfg = stream_->configuration();

    const FrameBuffer::Plane &plane = buffer->planes()[0];
    void *data = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE,
                      MAP_SHARED, plane.fd.get(), 0);

    if (data == MAP_FAILED)
        return;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // lastFrame_ = cv::Mat(
        //     cfg.size.height,
        //     cfg.size.width,
        //     CV_8UC3,
        //     data
        // ).clone();

        // RGB888 - simple copy with stride
        size_t stride = cfg.stride;
        lastFrame_ = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC3);
        
        uint8_t* src = static_cast<uint8_t*>(data);
        for (int y = 0; y < cfg.size.height; y++) {
            memcpy(lastFrame_.ptr(y), src + y * stride, cfg.size.width * 3);
        }

        frameReady_ = true;
    }

    munmap(data, plane.length);
    cv_.notify_one();

    if (running_) {
        request->reuse(Request::ReuseBuffers);
        camera_->queueRequest(request);
    }
}

cv::Mat LibCameraCapture::capture() {
    // static int skip = 5;
    // if (skip-- > 0)
    //     return cv::Mat();

    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]{ return frameReady_; });
    frameReady_ = false;
    // return lastFrame_.clone();

    cv::Mat frame = lastFrame_.clone();
    
    // Downscale if needed
    // if (targetWidth_ > 0 && targetHeight_ > 0 && 
    //     (frame.cols != targetWidth_ || frame.rows != targetHeight_)) {
    //     cv::Mat resized;
    //     cv::resize(frame, resized, cv::Size(targetWidth_, targetHeight_), 0, 0, cv::INTER_LINEAR);
    //     return resized;
    // }
    
    return frame;
}
