#ifndef PIECE_CROPPER_H
#define PIECE_CROPPER_H

#include <opencv2/opencv.hpp>
#include <vector>

// Utility for 3D/2D coordinate mapping
class CameraMapper {
public:
    static cv::Mat getK(cv::Size current_size, float f = 2739.79f, cv::Size original_size = cv::Size(4032, 3024));
    static void getExtrinsics(const cv::Mat& image_pts, const cv::Mat& K, cv::Mat& R, cv::Mat& t);
    static cv::Mat traceRay(const cv::Mat& points_2d, const cv::Mat& K, const cv::Mat& R_ext, float Z_target);
    static cv::Mat project3D(const cv::Mat& points_3d, const cv::Mat& K, const cv::Mat& R_ext);
};

// The primary class replacing the Python PieceCropper wrapper
class PieceCropper {
public:
    PieceCropper();
    
    // Generates the 3D grids and extracts the 64 squares
    std::vector<cv::Mat> process(const cv::Mat& img, const cv::Mat& corners);

private:
    cv::Mat grid_original_; // 81x2 matrix of warped-space points
};

// Global function kept for backward compatibility with your existing .cpp
std::vector<cv::Mat> extractWarpedSquares(
    const cv::Mat& image,
    const cv::Mat& grid_top,
    const cv::Mat& grid_bottom,
    int square_width = 64,
    int square_height = 128
);

#endif