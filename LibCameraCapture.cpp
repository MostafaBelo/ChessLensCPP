#include "LibCameraCapture.h"
#include <iostream>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <thread>
#include <chrono>

using namespace libcamera;

LibCameraCapture::LibCameraCapture(int width, int height) {
    cameraManager_ = std::make_unique<CameraManager>();
    cameraManager_->start();

    if (cameraManager_->cameras().empty())
        throw std::runtime_error("No cameras found");

    camera_ = cameraManager_->cameras()[0];
    camera_->acquire();

    // Add after camera_->acquire():
    std::cout << "Available stream configurations:" << std::endl;
    std::unique_ptr<CameraConfiguration> testConfig = 
        camera_->generateConfiguration({ StreamRole::Viewfinder });
    StreamConfiguration &sc = testConfig->at(0);
    std::cout << "  Default: " << sc.toString() << std::endl;

    std::unique_ptr<CameraConfiguration> config =
        camera_->generateConfiguration({ StreamRole::Viewfinder });

    StreamConfiguration &streamConfig = config->at(0);
    std::cout << "Default configuration is: " << streamConfig.toString() << std::endl;
    streamConfig.size.width = width;
    streamConfig.size.height = height;
    // streamConfig.pixelFormat = formats::YUV420;
    streamConfig.pixelFormat = formats::RGB888;
    streamConfig.bufferCount = 4;
    
    // config->transform = Transform::Identity;

    // Validate and let libcamera adjust if needed
    CameraConfiguration::Status status = config->validate();
    if (status == CameraConfiguration::Invalid) {
        throw std::runtime_error("Invalid camera configuration");
    }
    
    if (status == CameraConfiguration::Adjusted) {
        std::cout << "WARNING: Configuration was adjusted!" << std::endl;
        std::cout << "Requested: " << width << "x" << height << std::endl;
        std::cout << "Got: " << streamConfig.size.width << "x" << streamConfig.size.height << std::endl;
    }
    std::cout << "Final configuration is: " << streamConfig.toString() << std::endl;

    camera_->configure(config.get());
    stream_ = streamConfig.stream();

    allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
    allocator_->allocate(stream_);

    // Get the full sensor pixel array size
    auto pixelArrayOpt = camera_->properties().get(properties::PixelArraySize);
    auto activeAreasOpt = camera_->properties().get(properties::PixelArrayActiveAreas);

    Rectangle cropRegion;
    bool useCrop = false;
    
    if (pixelArrayOpt) {
        const Size &sensorSize = *pixelArrayOpt;
        std::cout << "Full sensor size: " << sensorSize.toString() << std::endl;
        
        // Use full sensor for wide-angle lens
        cropRegion = Rectangle(0, 0, sensorSize.width, sensorSize.height);
        useCrop = true;
        
        std::cout << "Setting crop to full sensor: " << cropRegion.toString() << std::endl;
    }
    for (const auto &buffer : allocator_->buffers(stream_)) {
        std::unique_ptr<Request> request = camera_->createRequest();

        // Enable AE + AWB
        request->controls().set(controls::AeEnable, true);
        request->controls().set(controls::AwbEnable, true);
        
        // Set to full sensor crop if available
        if (useCrop) {
            request->controls().set(controls::ScalerCrop, cropRegion);
        }

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
}

LibCameraCapture::~LibCameraCapture() {
    camera_->stop();
    camera_->release();
    cameraManager_->stop();
}

void LibCameraCapture::requestComplete(Request *request) {
    if (request->status() != Request::RequestComplete)
        return;

    const FrameBuffer *buffer = request->buffers().begin()->second;
    const FrameMetadata &metadata = buffer->metadata();
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

        // Handle different pixel formats
        if (cfg.pixelFormat == formats::YUV420) {
            // YUV420 has Y plane (full size) + U plane (1/4 size) + V plane (1/4 size)
            // Total height = height * 1.5
            cv::Mat yuvImage(cfg.size.height * 3 / 2, cfg.size.width, CV_8UC1, data);
            lastFrame_ = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC3);
            cv::cvtColor(yuvImage, lastFrame_, cv::COLOR_YUV420p2RGB);
        } else if (cfg.pixelFormat == formats::YUYV) {
            cv::Mat yuvImage(cfg.size.height, cfg.size.width, CV_8UC2, data);
            lastFrame_ = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC3);
            cv::cvtColor(yuvImage, lastFrame_, cv::COLOR_YUV2RGB_YUYV);
        } else if (cfg.pixelFormat == formats::RGB888) {
            lastFrame_ = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC3, data).clone();
        } else {
            std::cerr << "Unsupported pixel format: " << cfg.pixelFormat.toString() << std::endl;
        }

        frameReady_ = true;
    }

    munmap(data, plane.length);
    cv_.notify_one();

    request->reuse(Request::ReuseBuffers);
    camera_->queueRequest(request);
}

cv::Mat LibCameraCapture::capture() {
    // static int skip = 5;
    // if (skip-- > 0)
    //     return cv::Mat();

    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]{ return frameReady_; });
    frameReady_ = false;
    return lastFrame_.clone();
}
