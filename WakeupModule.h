#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <optional>

class WakeupModule {
public:
    WakeupModule();

    // img must be 256x256 CV_8UC3 (BGR or RGB – just be consistent)
    bool is_wakeup(const cv::Mat& img,
                   const std::optional<cv::Mat>& past_img = std::nullopt);

private:
    static constexpr int BINS = 8;
    static constexpr int GRID = 8;
    static constexpr int CELL = 32;

    using Hist = std::array<double, 3 * BINS>;
    using FrameHist = std::vector<Hist>; // size 64

    std::optional<FrameHist> past_frame_hist_;

    Hist compute_hist(const cv::Mat& img) const;
};
