#include "WakeupModule.h"
#include <cmath>
#include <algorithm>

WakeupModule::WakeupModule() : past_frame_hist_(std::nullopt) {}

WakeupModule::Hist WakeupModule::compute_hist(const cv::Mat& img) const {
    Hist hist{};
    hist.fill(0.0f);

    const int pixels = img.rows * img.cols;
    const double norm = 1.0f / pixels;
    const int bin_size = 256 / BINS;

    for (int y = 0; y < img.rows; ++y) {
        const cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; ++x) {
            const cv::Vec3b& p = row[x];
            for (int c = 0; c < 3; ++c) {
                int bin = std::min(p[c] / bin_size, BINS - 1);
                hist[c * BINS + bin] += 1.0f;
            }
        }
    }

    for (double& v : hist)
        v *= norm;

    return hist;
}

bool WakeupModule::is_wakeup(const cv::Mat& img,
                             const std::optional<cv::Mat>& past_img) {

    cv::Mat img_u8;
    if (img.type() != CV_8UC3) {
        img.convertTo(img_u8, CV_8UC3);
    } else {
        img_u8 = img;
    }

    FrameHist current_hist;
    current_hist.reserve(GRID * GRID);

    // split into 8x8 blocks
    for (int i = 0; i < GRID; ++i) {
        for (int j = 0; j < GRID; ++j) {
            cv::Rect roi(j * CELL, i * CELL, CELL, CELL);
            current_hist.push_back(compute_hist(img_u8(roi)));
        }
    }

    std::optional<FrameHist> past_hist;

    if (past_img.has_value()) {
        // single histogram like Python version (note: Python code seems inconsistent here,
        // but we replicate behavior)
        FrameHist ph;
        ph.push_back(compute_hist(*past_img));
        past_hist = ph;
    } else {
        past_hist = past_frame_hist_;
    }

    bool ret = true;

    if (past_hist.has_value()) {
        double max_mse = 0.0f;

        for (size_t i = 0; i < current_hist.size(); ++i) {
            const Hist& h1 = current_hist[i];
            const Hist& h2 = past_hist->size() == 1
                             ? (*past_hist)[0]
                             : (*past_hist)[i];

            double mse = 0.0f;
            for (size_t k = 0; k < h1.size(); ++k) {
                double d = h1[k] - h2[k];
                mse += d * d;
            }
            mse /= h1.size();
            max_mse = std::max(max_mse, mse);
        }

        double err = -std::log(max_mse + 1e-7f);
        ret = err < 6.0f;
    }

    if (ret) {
        past_frame_hist_ = current_hist;
    }

    return ret;
}
