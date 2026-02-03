#include "PieceDetection.h"

PieceDetector::PieceDetector(const std::string& model_path)
    : internal_detector_(std::make_unique<PieceDetectorCNN>(model_path)),
      cropper_(std::make_unique<PieceCropper>()) {}

PieceDetectorResult PieceDetector::process(const cv::Mat& img, const cv::Mat& corners) {
    // 1. Geometry & 3D Cropping
    // Uses the CameraMapper logic internally to return 64 Mat squares
    std::vector<cv::Mat> squares = cropper_->process(img, corners);

    // 2. Preprocess for ONNX: Convert vector<Mat> to flat float vector [1, 8, 8, 3, 128, 64]
    const int H = 128;
    const int W = 64;
    const int CH = 3;
    std::vector<float> board_split(1 * 8 * 8 * CH * H * W);

    // Strides for [1, 8, 8, 3, 128, 64]
    const int STRIDE_ROW = 8 * CH * H * W;
    const int STRIDE_COL = CH * H * W;
    const int STRIDE_CH  = H * W;

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            // Access the correct square from your vector
            // Assumes squares[0-7] is row 0, squares[8-15] is row 1...
            cv::Mat& square = squares[r * 8 + c]; 
            
            int base_offset = (r * STRIDE_ROW) + (c * STRIDE_COL);

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    cv::Vec3f pixel = square.at<cv::Vec3b>(h, w); 
                    
                    // Indexing based on [CH, H, W] planar format
                    board_split[base_offset + (0 * STRIDE_CH) + (h * W) + w] = pixel[0] / 255.0f; // R
                    board_split[base_offset + (1 * STRIDE_CH) + (h * W) + w] = pixel[1] / 255.0f; // G
                    board_split[base_offset + (2 * STRIDE_CH) + (h * W) + w] = pixel[2] / 255.0f; // B
                }
            }
        }
    }

    // 3. Predict using the ONNX session
    return internal_detector_->predict(board_split, H, W);
}