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

    std::unique_ptr<CameraConfiguration> config =
        camera_->generateConfiguration({ StreamRole::StillCapture });

    StreamConfiguration &streamConfig = config->at(0);
    std::cout << "Default viewfinder configuration is: " << streamConfig.toString() << std::endl;
    streamConfig.size.width = width;
    streamConfig.size.height = height;
    streamConfig.pixelFormat = formats::RGB888;
    streamConfig.bufferCount = 4;
    std::cout << "Adjusted viewfinder configuration is: " << streamConfig.toString() << std::endl;

    // config->transform = Transform::Identity;

    if (config->validate() == CameraConfiguration::Invalid)
        throw std::runtime_error("Invalid camera configuration");

    camera_->configure(config.get());
    stream_ = streamConfig.stream();

    allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
    allocator_->allocate(stream_);

    // auto pixelArray = camera_->properties().get(libcamera::properties::PixelArraySize);
    // const libcamera::Size &outputSize = streamConfig.size;

    // libcamera::Rectangle crop(
    //     (pixelArray->width  - outputSize.width)  / 2,
    //     (pixelArray->height - outputSize.height) / 2,
    //     outputSize.width,
    //     outputSize.height
    // );

    for (const auto &buffer : allocator_->buffers(stream_)) {
        std::unique_ptr<Request> request = camera_->createRequest();

        // Enable AE + AWB (CRITICAL)
        request->controls().set(libcamera::controls::AeEnable, true);
        request->controls().set(libcamera::controls::AwbEnable, true);

        // request->controls().set(libcamera::controls::ScalerCrop, crop);

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
        lastFrame_ = cv::Mat(
            cfg.size.height,
            cfg.size.width,
            CV_8UC3,
            data
        ).clone();
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
