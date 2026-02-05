#include "PieceDetection.h"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <thread>

void printBoardCNN(std::vector<float> board) {
    // Piece labels mapping to the 13 channels
    const char* symbols[] = {
        "P", "N", "B", "R", "Q", "K", // White (0-5)
        "p", "n", "b", "r", "q", "k", // Black (6-11)
        "."                           // Empty (12)
    };

    std::cout << "\n--- Detected Board (Max Channel per Square) ---\n";
    for (int r = 0; r < 8; ++r) {
        std::cout << (8 - r) << " "; // Rank labels
        for (int c = 0; c < 8; ++c) {
            int pos = r * 8 + c;
            
            int max_idx = 0;
            float max_val = -1.0f;

            // Iterate through the 13 channels for this specific square
            for (int ch = 0; ch < 13; ++ch) {
                float val = board[ch * 64 + pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = ch;
                }
            }
            std::cout << symbols[max_idx] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "  a b c d e f g h\n\n";

    for (int r = 0; r < 8; ++r) {
        std::cout << (8 - r) << " "; // Rank labels
        for (int c = 0; c < 8; ++c) {
            int pos = r * 8 + c;
            
            int max_idx = 0;
            float max_val = -1.0f;

            // Iterate through the 13 channels for this specific square
            for (int ch = 0; ch < 13; ++ch) {
                float val = board[ch * 64 + pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = ch;
                }
            }
            std::cout << max_val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "  a b c d e f g h\n\n";
}

PieceDetectorCNN::PieceDetectorCNN(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PieceDetectorCNN"),
      session_(nullptr),  // Initialize with nullptr first
      opts_() {
    
    // opts_.SetIntraOpNumThreads(1);
    // opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    int num_threads = std::thread::hardware_concurrency();
    opts_.SetIntraOpNumThreads(num_threads);
    opts_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Create session after options are set
    session_ = Ort::Session(env_, onnx_path.c_str(), opts_);
    
    // Get input/output names (ONNX Runtime 1.23.2+ API)
    auto input_names = session_.GetInputNames();
    if (!input_names.empty()) {
        input_name_ = input_names[0];
    }
    
    // Get output names
    auto output_names = session_.GetOutputNames();
    output_names_.clear();
    for (const auto& name : output_names) {
        output_names_.push_back(name);
    }
}

PieceDetectorResult PieceDetectorCNN::predict(const std::vector<float>& board_split,
                                          int H, int W) {
    // Input shape: [1, 8, 8, 3, H, W]
    std::vector<int64_t> input_shape = {1, 8, 8, 3, H, W};
    
    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(board_split.data()),
        board_split.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    // Prepare input/output names as const char* arrays
    const char* input_names_cstr[] = {input_name_.c_str()};
    
    std::vector<const char*> output_names_cstr;
    for (const auto& name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }
    
    // Run inference
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_cstr,
        &input_tensor,
        1,
        output_names_cstr.data(),
        output_names_cstr.size()
    );
    
    // Extract outputs
    // Assuming model outputs: [occupancy, color, type]
    // occupancy: [1, 8, 8]
    // color: [1, 8, 8]
    // type: [1, 8, 8, 6]
    
    float* occ = outputs[0].GetTensorMutableData<float>();       // [1,8,8]
    float* color = outputs[1].GetTensorMutableData<float>();     // [1,8,8]
    float* type = outputs[2].GetTensorMutableData<float>();      // [1,8,8,6]
    
    // Build final board [13, 8, 8]
    // Channels: P, N, B, R, Q, K, p, n, b, r, q, k, empty
    std::vector<float> board(13 * 8 * 8, 0.0f);
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int pos = r * 8 + c;
            
            float occ_prob = sigmoid(occ[pos]);
            float color_prob = sigmoid(color[pos]);  // 0=black, 1=white
            
            // Get piece type probabilities
            float type_probs[6];
            for (int t = 0; t < 6; ++t) {
                type_probs[t] = type[pos * 6 + t];
            }
            softmax(type_probs, 6);

            board[12 * 64 + pos] = 1-occ_prob;
            for (int t = 0; t < 6; ++t) {
                board[t * 64 + pos] = type_probs[t] * color_prob * occ_prob;
            }
            for (int t = 0; t < 6; ++t) {
                board[(t+6) * 64 + pos] = type_probs[t] * (1-color_prob) * occ_prob;
            }
        }
    }
    
    PieceDetectorResult result;
    result.board = board;
    return result;
}

float PieceDetectorCNN::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void PieceDetectorCNN::softmax(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        data[i] = std::exp(data[i]);
        sum += data[i];
    }
    
    for (int i = 0; i < n; ++i) {
        data[i] /= sum;
    }
}