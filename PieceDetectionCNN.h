#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * Result structure for piece detection
 */
struct PieceDetectorResult {
    // Final board tensor: [13, 8, 8] flattened
    // Channels: P, N, B, R, Q, K, p, n, b, r, q, k, empty
    std::vector<float> board;
};

/**
 * Chess piece detector using ONNX Runtime
 * 
 * Input format:
 *   board_split: flattened vector representing [1, 8, 8, 3, H, W]
 *   Layout: [batch=1][rank=8][file=8][channel=3][height=H][width=W]
 * 
 * Output format:
 *   board: flattened vector representing [13, 8, 8]
 *   Layout: [piece_channel=13][rank=8][file=8]
 */
class PieceDetectorCNN {
public:
    /**
     * Constructor
     * @param onnx_path Path to ONNX model file
     */
    explicit PieceDetectorCNN(const std::string& onnx_path);

    /**
     * Predict piece positions
     * 
     * @param board_split Input tensor [1, 8, 8, 3, H, W] flattened
     * @param H Height of each square image
     * @param W Width of each square image
     * @return Detection result with board probabilities
     */
    PieceDetectorResult predict(const std::vector<float>& board_split,
                                int H, int W);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::SessionOptions opts_;

    std::string input_name_;
    std::vector<std::string> output_names_;

    static float sigmoid(float x);
    static void softmax(float* data, int n);
};