#ifndef BOARD_DETECTION_H
#define BOARD_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "BoardSaddle.h" // Assuming the previous files are named this

class BoardExtractor {
public:
    BoardExtractor();

    /**
     * Replicates the Python extract_board.
     * Takes an image, detects corners, and orders them rotation-proof.
     * Returns a 4x2 CV_32F Matrix.
     */
    cv::Mat extractBoard(const cv::Mat& inputImg);

    /**
     * Replicates the Python warp.
     * Warps the image based on the detected quad.
     */
    cv::Mat warp(const cv::Mat& img, const cv::Mat& quad);
    std::pair<cv::Mat, cv::Mat> warp(const cv::Mat& img, const cv::Mat& quad, cv::Size target_size = cv::Size(256, 256));

private:
    /**
     * The C++ implementation of _order_points_rotation_proof
     */
    cv::Mat _orderPointsRotationProof(const cv::Mat& pts);
};

#endif