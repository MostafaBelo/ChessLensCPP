#include "ChessLens.h"
#include "Utils/ChessUtils.h"
#include <iostream>
#include <fstream>
// #include <span>
#include <algorithm>
#include <cmath>

void printBoard(const std::vector<float>& board) {
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
            // int pos = r * 8 + c;
            int pos = r * 8 * 13 + c * 13;
            
            int best_idx = 0;
            float best_val = 1000;

            // Iterate through the 13 channels for this specific square
            for (int ch = 0; ch < 13; ++ch) {
                // float val = board[ch * 64 + pos];
                float val = board[pos + ch];
                if (val < best_val) {
                    best_val = val;
                    best_idx = ch;
                }
            }
            std::cout << symbols[best_idx] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "  a b c d e f g h\n\n";

    for (int r = 0; r < 8; ++r) {
        std::cout << (8 - r) << " "; // Rank labels
        for (int c = 0; c < 8; ++c) {
            // int pos = r * 8 + c;
            int pos = r * 8 * 13 + c * 13;
            
            int best_idx = 0;
            float best_val = 1000;

            // Iterate through the 13 channels for this specific square
            for (int ch = 0; ch < 13; ++ch) {
                // float val = board[ch * 64 + pos];
                float val = board[pos + ch];
                if (val < best_val) {
                    best_val = val;
                    best_idx = ch;
                }
            }
            std::cout << best_val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "  a b c d e f g h\n\n";
}

// ============================================================================
// AvgTimes Implementation
// ============================================================================

void AvgTimes::print() const {
    if (count == 0) {
        std::cout << "No timing data available\n";
        return;
    }
    
    std::cout << "Average Times (ms) over " << count << " frames:\n";
    std::cout << "  Load:               " << (load / count) * 1000.0 << "\n";
    std::cout << "  Board Detection:    " << (board_detection / count) * 1000.0 << "\n";
    std::cout << "  Wakeup:             " << (wakeup / count) * 1000.0 << "\n";
    std::cout << "  Occlusion:          " << (occlusion / count) * 1000.0 << "\n";
    std::cout << "  Piece Recognition:  " << (piece_recognition / count) * 1000.0 << "\n";
    std::cout << "  HMM:                " << (hmm / count) * 1000.0 << "\n";
}

void AvgTimes::reset() {
    load = 0.0;
    board_detection = 0.0;
    wakeup = 0.0;
    occlusion = 0.0;
    piece_recognition = 0.0;
    hmm = 0.0;
    count = 0;
}

// ============================================================================
// ChessLensImage Implementation
// ============================================================================

ChessLensImage::ChessLensImage(const std::string& piece_detector_path,
                               const std::string& occlusion_detector_path)
    : board_extractor_(std::make_unique<BoardExtractor>()),
      wakeup_module_(std::make_unique<WakeupModule>()),
      occlusion_detector_(std::make_unique<OcclusionDetector>(occlusion_detector_path)),
      piece_detector_(std::make_unique<PieceDetector>(piece_detector_path)) {
    clear();
}

void ChessLensImage::clear() {
    img_ = cv::Mat();
    warped_img_ = cv::Mat();
    board_corners_ = cv::Mat();
    M_ = cv::Mat();
    piece_matrix_.clear();
    fen_.clear();
    board_detected_ = false;
    pieces_detected_ = false;
}

void ChessLensImage::load_image(const cv::Mat& img) {
    clear();
    img_ = prep_img(img);
}

void ChessLensImage::load_image(const std::string& img_path) {
    clear();
    cv::Mat loaded = cv::imread(img_path, cv::IMREAD_COLOR);
    if (loaded.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path);
    }
    img_ = prep_img(loaded);
}

cv::Mat ChessLensImage::prep_img(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));
    return resized;
}

std::pair<cv::Mat, float> ChessLensImage::detect_board(bool verbose) {
    if (!is_img_loaded()) {
        throw std::runtime_error("No image loaded");
    }
    
    cv::Mat result = board_extractor_->extractBoard(img_);
    // auto result = detectChessboardCorners(img_, board_config_);
    
    if (result.empty()) {
        throw std::runtime_error("Board detection failed");
    }
    
    board_corners_ = result.clone();
    board_detected_ = true;
    
    // Placeholder confidence (board detection doesn't return confidence)
    float conf = 1.0f;
    
    return {board_corners_, conf};
}

std::pair<cv::Mat, cv::Mat> ChessLensImage::warp() {
    if (!is_board_detected()) {
        throw std::runtime_error("Board not detected");
    }
    
    if (!warped_img_.empty()) {
        return {warped_img_, M_};
    }
    
    // Use the board_extractor logic (target size 256x256 as requested)
    auto result = board_extractor_->warp(img_, board_corners_, cv::Size(256, 256));
    warped_img_ = result.first;
    M_ = result.second;
    
    cv::imwrite("warped.png", warped_img);
    return {warped_img_, M_};
}

bool ChessLensImage::is_wakeup() {
    auto [warped, _] = warp();
    return wakeup_module_->is_wakeup(warped);
}

bool ChessLensImage::is_occluded() {
    auto [warped, _] = warp();
    auto [is_occ, conf] = occlusion_detector_->is_occluded(warped);
    return is_occ;
}

std::pair<std::vector<float>, std::string> ChessLensImage::recognize_pieces(bool verbose) {
    if (!is_img_loaded() || !is_board_detected()) {
        throw std::runtime_error("Image not loaded or board not detected");
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    PieceDetectorResult result = piece_detector_->process(img_, board_corners_);
    
    // Map output [13, 8, 8] to internal [8, 8, 13]
    piece_matrix_.assign(8 * 8 * 13, 0.0f);
    for (int ch = 0; ch < 13; ++ch) {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                // CNN format: [channel][row][col]
                // State format: [row][col][channel]
                piece_matrix_[r * 8 * 13 + c * 13 + ch] = result.board[ch * 64 + r * 8 + c];
            }
        }
    }
    
    fen_ = ChessUtils::tensor_to_fen_max(piece_matrix_);
    pieces_detected_ = true;
    
    if (verbose) {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Piece Recognition (Encapsulated 3D): " << ms << " ms\n";
    }
    
    return {piece_matrix_, fen_};
}

cv::Mat ChessLensImage::get_fen_img() const {
    if (!is_pieces_detected()) {
        throw std::runtime_error("Pieces not detected");
    }
    return ChessUtils::fen_to_png(fen_);
}

void ChessLensImage::save_fen_image(const std::string& file_name) const {
    cv::Mat fen_img = get_fen_img();
    if (!fen_img.empty()) {
        cv::imwrite(file_name, fen_img);
    }
}

// ============================================================================
// ChessLensGame1 Implementation
// ============================================================================

ChessLensGame1::ChessLensGame1(const ChessLensConfig& config,
                               const std::string& piece_detector_path,
                               const std::string& occlusion_detector_path)
    : board_flag(false), config_(config) {
    
    camera_ = std::make_unique<ImageProvider>(
        CameraType::PI_FISH, config.camera_interval);
    
    current_img_ = std::make_unique<ChessLensImage>(
        piece_detector_path, occlusion_detector_path);
    
    clear();
}

void ChessLensGame1::clear() {
    board_detection_ = cv::Mat();
    board_fails_count_ = 0;
    t_ = 0;
    last_wakeup_ = 0;
}

std::vector<float> ChessLensGame1::operate() {
    cv::Mat img = camera_->take_image();
    if (img.empty()) {
        return {};
    }
    return set_img(img);
}

std::vector<float> ChessLensGame1::set_img(const cv::Mat& img) {
    auto t1 = std::chrono::high_resolution_clock::now();

    current_img_->load_image(img);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto result = process_img();
    
    t_++;
    
    auto t3 = std::chrono::high_resolution_clock::now();
    
    avg_times.load += std::chrono::duration<double>(t2 - t1).count();
    avg_times.count++;
    
    return result;
}

bool ChessLensGame1::detect_wakeup() {
    return current_img_->is_wakeup();
}

bool ChessLensGame1::detect_occlusion() {
    return current_img_->is_occluded();
}

std::vector<float> ChessLensGame1::prep_probs(const std::vector<float>& probs) {
    // probs is [8, 8, 13], just return as-is
    return probs;
}

std::vector<float> ChessLensGame1::process_img() {
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Board Detection
    if (t_ % config_.bd_period == 0) {
        try {
            auto [new_corners, conf] = current_img_->detect_board();
            
            if (board_detection_.empty()) {
                board_detection_ = new_corners.clone();
            } else {
                // Average with previous detection
                board_detection_ = (new_corners + board_detection_) / 2.0;
            }
            
            board_fails_count_ = 0;
            board_flag.set(false);
        } catch (const std::exception& e) {
            board_fails_count_++;
            // cout << "board_fails_count " << board_fails_count_ << " board_flag " << board_flag.get() << " max_bd_fails " << config_.max_bd_fails;
            if (board_fails_count_ >= config_.max_bd_fails || board_detection_.empty()) {
                board_flag.set(true);
                return {};
            }
        }
    }
    
    current_img_->board_corners_ = board_detection_;
    current_img_->board_detected_ = !board_detection_.empty();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Wakeup Detection
    bool is_wakeup;
    if (last_wakeup_ - t_ >= config_.wakeup_period) {
        is_wakeup = true;
    } else {
        is_wakeup = detect_wakeup();
    }
    
    cout << "Wakeup: " << (is_wakeup ? "True" : "False") << "\n";
    if (!is_wakeup) {
        return {};
    }
    last_wakeup_ = t_;
    
    auto t3 = std::chrono::high_resolution_clock::now();
    
    // Occlusion Detection
    bool is_occluded = detect_occlusion();
    cout << "Occlusion: " << (is_occluded ? "True" : "False") << "\n";
    if (is_occluded) {
        return {};
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    
    // Piece Recognition
    auto [piece_matrix, fen] = current_img_->recognize_pieces();
    auto probs = prep_probs(piece_matrix);
    
    auto t5 = std::chrono::high_resolution_clock::now();
    
    avg_times.board_detection += std::chrono::duration<double>(t2 - t1).count();
    avg_times.wakeup += std::chrono::duration<double>(t3 - t2).count();
    avg_times.occlusion += std::chrono::duration<double>(t4 - t3).count();
    avg_times.piece_recognition += std::chrono::duration<double>(t5 - t4).count();
    
    return probs;
}

void ChessLensGame1::quit() {
    camera_->quit();
}

// ============================================================================
// ChessLensGame2 Implementation
// ============================================================================

ChessLensGame2::ChessLensGame2(const ChessLensConfig& config)
    : config_(config),
      latest_bound_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR") {
    
    clear();
}

void ChessLensGame2::clear() {
    orientation_ = Orientation::UNKNOWN;
    context_model_ = std::make_unique<ContextAwareModels::HMM>(
        config_.context_breadth, 
        config_.context_delay, 
        config_.context_bind_period
    );
}

void ChessLensGame2::calc_orientation(const std::vector<float>& piece_matrix) {
    // piece_matrix is [8, 8, 13] - find argmax
    std::vector<int> board(64);
    for (int i = 0; i < 64; ++i) {
        int max_idx = 0;
        float max_val = piece_matrix[i * 13];
        for (int ch = 1; ch < 13; ++ch) {
            if (piece_matrix[i * 13 + ch] > max_val) {
                max_val = piece_matrix[i * 13 + ch];
                max_idx = ch;
            }
        }
        board[i] = max_idx;
    }
    
    // Count whites (pieces < 6) in different regions
    int whites_count[64] = {0};
    for (int i = 0; i < 64; ++i) {
        whites_count[i] = (board[i] < 6) ? 1 : 0;
    }
    
    int correct_r = 0, correct_l = 0, correct_t = 0, correct_b = 0;
    
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int idx = r * 8 + c;
            if (c >= 4) correct_r += whites_count[idx];
            else correct_r -= whites_count[idx];
            
            if (r < 4) correct_t += whites_count[idx];
            else correct_t -= whites_count[idx];
        }
    }
    
    correct_l = -correct_r;
    correct_b = -correct_t;
    
    int vals[] = {correct_r, correct_l, correct_t, correct_b};
    int max_idx = std::max_element(vals, vals + 4) - vals;
    
    Orientation orientations[] = {Orientation::RIGHT, Orientation::LEFT, 
                                  Orientation::TOP, Orientation::BOTTOM};
    orientation_ = orientations[max_idx];
    
    std::string orient_str[] = {"RIGHT", "LEFT", "TOP", "BOTTOM"};
    std::cout << "Orientation: " << orient_str[max_idx] << "\n";
}

std::vector<float> ChessLensGame2::prep_probs(const std::vector<float>& probs) {
    // Rotate and apply -log
    // probs is [8, 8, 13]
    std::vector<float> result(8 * 8 * 13);
    
    int k = 0;
    if (orientation_ == Orientation::RIGHT) k = 3;  // rot90 k=-1 equivalent
    else if (orientation_ == Orientation::LEFT) k = 1;
    else if (orientation_ == Orientation::TOP) k = 2;
    else if (orientation_ == Orientation::BOTTOM) k = 0;
    
    // Apply rotation and -log transform
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int new_r = r, new_c = c;
            
            // Apply k rotations (90° counterclockwise each)
            for (int rot = 0; rot < k; ++rot) {
                int tmp = new_r;
                new_r = new_c;
                new_c = 7 - tmp;
            }
            
            // Flip vertically ([::-1])
            new_r = 7 - new_r;
            
            for (int ch = 0; ch < 13; ++ch) {
                float val = probs[r * 8 * 13 + c * 13 + ch];
                // float res = -std::log(val + 1e-7f);
                result[new_r * 8 * 13 + new_c * 13 + ch] = -std::log(val + 1e-7f);
                // result[new_r * 8 * 13 + new_c * 13 + ch] = val;
                
                // cout << "val: " << res << "\n";
                // if (res < 0) cout << "val: " << res << "\n";
            }
        }
    }
    
    return result;
}

void ChessLensGame2::operate(const std::vector<float>& piece_matrix) {
    if (orientation_ == Orientation::UNKNOWN) {
        calc_orientation(piece_matrix);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // printBoard(piece_matrix);
    auto prepped = prep_probs(piece_matrix);
    // printBoard(prepped);

    int timestep = context_model_->top_t() + 1;
    
    timestamp_map_[timestep] = std::chrono::steady_clock::now();
    context_model_->set_probs(timestep, prepped, timestamp_map_[timestep]);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    avg_times.hmm += std::chrono::duration<double>(t2 - t1).count();
    avg_times.count++;
}

void ChessLensGame2::update_bindings() {
    auto now = std::chrono::steady_clock::now();
    
    bool isbound = context_model_->check_bind(now);
    
    if (isbound || config_.context_continuous) {
        get_latest_fens();
    }
}

void ChessLensGame2::bind() {
    if (context_model_->top_t() != context_model_->top_bind_t()) {
        context_model_->bind();
    }
}

std::vector<std::string> ChessLensGame2::get_history(bool include_non_bound) {
    auto hist = context_model_->get_history(include_non_bound);
    
    std::vector<std::string> fens;
    for (int i = 0; i < hist.shape.i; i++)
    {
        // fens.push_back(tensor_to_fen(std::span(hist.data_.begin() + (i*64), (i+1)*64)));

        std::vector<int> span(hist.data_.begin() + (i*64), hist.data_.begin() + (i+1)*64);
        fens.push_back(ChessUtils::tensor_to_fen(span));
    }

    std::vector<std::string> new_fens;

    for (int i = 0; i < fens.size(); i++)
    {
        if ((i == 0) || (fens[i] != fens[i-1])) new_fens.push_back(fens[i] + " - " + to_string(fens.size()));
    }
    
    return fens;
}

void ChessLensGame2::get_latest_fens() {
    auto fens = get_history(config_.context_continuous);
    
    // Remove duplicates and already broadcasted FENs
    std::vector<std::string> new_fens;
    for (const auto& fen : fens) {
        if (std::find(broadcasted_fens_.begin(), broadcasted_fens_.end(), fen) 
            == broadcasted_fens_.end()) {
            new_fens.push_back(fen);
            broadcasted_fens_.push_back(fen);
        }
    }
    
    std::cout << "FENS: ";
    for (const auto& fen : new_fens) {
        std::cout << fen << " ";
    }
    std::cout << "\n";
    
    if (config_.fen_update && !new_fens.empty()) {
        latest_bound_fen = new_fens.back();
        config_.fen_update(latest_bound_fen);
    }
    
    if (!config_.game_out_path.empty() && !new_fens.empty()) {
        std::ofstream out(config_.game_out_path + "/game_fens.csv", std::ios::app);
        for (const auto& fen : new_fens) {
            out << fen << "\n";
        }
    }
}

void ChessLensGame2::quit() {
    // Cleanup if needed
}

// ============================================================================
// ChessLensGame Implementation
// ============================================================================

ChessLensGame::ChessLensGame(const ChessLensConfig& config,
                             const std::string& piece_detector_path,
                             const std::string& occlusion_detector_path)
    : config_(config) {
    
    game1_ = std::make_unique<ChessLensGame1>(config, piece_detector_path, 
                                              occlusion_detector_path);
    game2_ = std::make_unique<ChessLensGame2>(config);
}

void ChessLensGame::clear() {
    game1_->clear();
    game2_->clear();
}

bool ChessLensGame::operate(bool verbose) {
    auto probs = game1_->operate();
    
    if (probs.empty()) {
        return false;
    }
    
    game2_->operate(probs);
    game2_->update_bindings();
    
    return true;
}

bool ChessLensGame::set_img(const cv::Mat& img, bool verbose) {
    auto probs = game1_->set_img(img);
    
    if (probs.empty()) {
        return false;
    }
    
    game2_->operate(probs);
    game2_->update_bindings();
    
    return true;
}

void ChessLensGame::bind() {
    game2_->bind();
}

std::vector<std::string> ChessLensGame::get_history(bool include_non_bound) {
    return game2_->get_history(include_non_bound);
}

void ChessLensGame::quit() {
    game1_->quit();
    game2_->quit();
}

AvgTimes ChessLensGame::get_combined_avg_times() const {
    AvgTimes combined = game1_->avg_times;
    combined.hmm = game2_->avg_times.hmm;
    return combined;
}
