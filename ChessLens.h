#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <chrono>
#include <map>

#include "ImageProvider.h"
#include "BoardDetection.h"
#include "WakeupModule.h"
#include "OcclusionDetector.h"
#include "PieceDetection.h"
#include "ContextAwareModels/HMM.h"
#include "Utils/Utils.h"

/**
 * Configuration structure for ChessLens system
 */
struct ChessLensConfig {
    // Camera settings
    double camera_interval = 0.2;
    
    // Board detection
    int bd_period = 5;              // Detect board every N frames
    int max_bd_fails = 5;           // Max consecutive failures before flagging
    
    // Wakeup detection
    int wakeup_period = 10;         // Minimum frames between wakeup checks
    
    // Context model settings
    int context_breadth = 50;       // Max width of HMM search tree
    double context_delay = 120.0;   // Delay in seconds before binding
    int context_bind_period = 1;    // Binding check period
    bool context_continuous = false; // Include non-bound states in history
    
    // Feature flags
    bool is_detect_occlusion = true;
    bool is_detect_wakeup = true;
    
    // Output settings
    std::string game_out_path = "";
    std::function<void(const std::string&)> fen_update = nullptr;
};

/**
 * Average timing statistics
 */
struct AvgTimes {
    double img_capture = 0.0;
    double load = 0.0;
    double board_detection = 0.0;
    double wakeup = 0.0;
    double occlusion = 0.0;
    double piece_recognition = 0.0;
    double hmm = 0.0;

    int img_capture_count = 0;
    int load_count = 0;
    int board_count = 0;
    int wakeup_count = 0;
    int occlusion_count = 0;
    int piece_count = 0;
    int hmm_count = 0;
    
    int count = 0;
    
    void print() const;
    void reset();
};

/**
 * Board orientation relative to camera
 */
enum class Orientation {
    UNKNOWN,
    RIGHT,   // White on right (rotate -90°)
    LEFT,    // White on left (rotate 90°)
    TOP,     // White on top (rotate 180°)
    BOTTOM   // White on bottom (no rotation)
};

/**
 * Single chess image processor
 * Handles board detection, warping, and piece recognition for one image
 */
class ChessLensImage {
public:
    ChessLensImage(const std::string& piece_detector_path,
                   const std::string& occlusion_detector_path);
    
    void clear();
    
    // State queries
    bool is_img_loaded() const { return !img_.empty(); }
    bool is_board_detected() const { return board_detected_; }
    bool is_pieces_detected() const { return pieces_detected_; }
    
    // Image loading
    void load_image(const cv::Mat& img);
    void load_image(const std::string& img_path);
    
    // Processing pipeline
    std::pair<cv::Mat, float> detect_board(bool verbose = false);
    std::pair<cv::Mat, cv::Mat> warp();
    bool is_wakeup();
    bool is_occluded();
    std::pair<std::vector<float>, std::string> recognize_pieces(bool verbose = false);
    
    // Output
    cv::Mat get_fen_img() const;
    void save_fen_image(const std::string& file_name = "out_fen.png") const;
    
    // Public members for direct access
    cv::Mat img_;
    cv::Mat warped_img_;
    cv::Mat board_corners_;  // 4x2 CV_32F
    cv::Mat M_;              // Perspective transform matrix
    std::vector<float> piece_matrix_;  // [8, 8, 13] probabilities
    std::string fen_;

    bool board_detected_ = false;
    bool pieces_detected_ = false;

private:
    std::unique_ptr<BoardExtractor> board_extractor_;
    std::unique_ptr<WakeupModule> wakeup_module_;
    std::unique_ptr<OcclusionDetector> occlusion_detector_;
    std::unique_ptr<PieceDetector> piece_detector_;
    
    ChessboardDetectionConfig board_config_;
    
    cv::Mat prep_img(const cv::Mat& img);
};

/**
 * ChessLensGame1: Image processing pipeline
 * Handles camera input, board detection, and piece recognition
 * Filters frames using wakeup and occlusion detection
 */
class ChessLensGame1 {
public:
    ChessLensGame1(const ChessLensConfig& config,
                   const std::string& piece_detector_path,
                   const std::string& occlusion_detector_path);
    
    void clear();
    
    // Main processing
    std::vector<float> operate();  // Process next camera frame
    std::vector<float> set_img(const cv::Mat& img);  // Process specific image
    
    void quit();
    
    // Observable flag for board detection failures
    Utils::Observable<bool> board_flag;
    
    // Performance tracking
    AvgTimes avg_times;

private:
    std::unique_ptr<ImageProvider> camera_;
    std::unique_ptr<ChessLensImage> current_img_;
    
    cv::Mat board_detection_;  // Averaged board corners
    int board_fails_count_ = 0;
    
    int t_ = 0;  // Frame counter
    int last_wakeup_ = 0;
    
    ChessLensConfig config_;
    
    bool detect_wakeup();
    bool detect_occlusion();
    std::vector<float> prep_probs(const std::vector<float>& probs);
    std::vector<float> process_img();
};

/**
 * ChessLensGame2: Context-aware game tracking
 * Uses HMM to track game state and handle orientation
 * Manages FEN history and time-delayed binding
 */
class ChessLensGame2 {
public:
    explicit ChessLensGame2(const ChessLensConfig& config);
    
    void clear();
    
    // Processing
    void operate(const std::vector<float>& piece_matrix);
    void update_bindings();
    void bind();
    
    // History and output
    std::vector<std::string> get_history(bool include_non_bound = false);
    void get_latest_fens();
    
    void quit();
    
    // Performance tracking
    AvgTimes avg_times;
    
    // Latest bound FEN position
    std::string latest_bound_fen;

private:
    std::unique_ptr<ContextAwareModels::HMM> context_model_;
    
    Orientation orientation_ = Orientation::UNKNOWN;
    
    ChessLensConfig config_;
    
    std::vector<std::string> broadcasted_fens_;
    std::map<int, std::chrono::steady_clock::time_point> timestamp_map_;
    
    void calc_orientation(const std::vector<float>& piece_matrix);
    std::vector<float> prep_probs(const std::vector<float>& probs);
};

/**
 * ChessLensGame: Combined full pipeline
 * Integrates Game1 (image processing) and Game2 (context tracking)
 * Provides unified interface for complete chess game tracking
 */
class ChessLensGame {
public:
    ChessLensGame(const ChessLensConfig& config,
                  const std::string& piece_detector_path,
                  const std::string& occlusion_detector_path);
    
    void clear();
    
    // Main processing loop
    bool operate(bool verbose = false);  // Process next camera frame
    bool set_img(const cv::Mat& img, bool verbose = false);  // Process specific image
    
    // Game state management
    void bind();
    std::vector<std::string> get_history(bool include_non_bound = false);
    
    void quit();
    
    // Performance tracking
    AvgTimes get_combined_avg_times() const;

private:
    std::unique_ptr<ChessLensGame1> game1_;
    std::unique_ptr<ChessLensGame2> game2_;
    
    ChessLensConfig config_;
};