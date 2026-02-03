#ifndef PIECE_DETECTOR_WRAPPER_H
#define PIECE_DETECTOR_WRAPPER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include "PieceDetectionCNN.h"
#include "PieceCropper.h"

class PieceDetector {
public:
    explicit PieceDetector(const std::string& model_path);

    // This handles the 3D grid generation, cropping, and ONNX prediction
    PieceDetectorResult process(const cv::Mat& img, const cv::Mat& corners);

private:
    std::unique_ptr<PieceCropper> cropper_;
    std::unique_ptr<PieceDetectorCNN> internal_detector_;
};

#endif