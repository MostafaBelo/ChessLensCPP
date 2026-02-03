#include "ChessLens.h"
#include "Utils/ChessUtils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <numeric>
#include <csignal>
#include <atomic>

// For HTTP requests (optional - can be disabled if not needed)
// You'll need a library like libcurl or cpp-httplib
#ifdef USE_HTTP
#include <curl/curl.h>
#endif

// Global flag for signal handling
std::atomic<bool> is_running(true);

/**
 * Signal handler for clean shutdown (Ctrl+C)
 */
void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nStopped Manually\n";
        is_running = false;
    }
}

/**
 * Update FEN via HTTP request
 * Requires libcurl or similar HTTP library
 */
void update_fen(const std::string& fen) {
#ifdef USE_HTTP
    CURL* curl = curl_easy_init();
    if (curl) {
        // std::string url = "http://10.42.0.0:8000/update_fen?fen=" + fen;
        std::string url = "http://localhost:8000/update_fen?fen=" + fen;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L); // HEAD request
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " 
                      << curl_easy_strerror(res) << std::endl;
        }
        curl_easy_cleanup(curl);
    }
#else
    // HTTP disabled - just print the FEN
    std::cout << "FEN Update: " << fen << std::endl;
#endif
}

/**
 * Calculate average of a vector
 */
template<typename T>
double calculate_average(const std::vector<T>& vec) {
    if (vec.empty()) return 0.0;
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

int main(int argc, char** argv) {
    // Setup signal handler for Ctrl+C
    std::signal(SIGINT, signal_handler);
    
#ifdef USE_HTTP
    // Initialize curl (if using HTTP)
    curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
    
    try {
        // Configuration
        std::string algorithm = "cnn_onnx_static";
        std::string dirname = "game_fens";
        
        // Parse command line arguments (optional)
        if (argc > 1) {
            algorithm = argv[1];
        }
        if (argc > 2) {
            dirname = argv[2];
        }
        
        // Model paths - adjust these to your actual paths
        std::string piece_detector_path = "models/" + algorithm + ".onnx";
        std::string occlusion_detector_path = "models/occlusion_detector.onnx";
        
        // Configure ChessLens
        ChessLensConfig config;
        config.camera_interval = 1;
        config.context_delay = 5.0;  // 5 seconds
        config.context_continuous = true;
        config.game_out_path = dirname;
        config.fen_update = update_fen;
        
        // Create game instances
        std::cout << "Initializing ChessLens...\n";
        std::cout << "Algorithm: " << algorithm << "\n";
        std::cout << "Output directory: " << dirname << "\n";
        
        ChessLensGame1 game1(config, piece_detector_path, occlusion_detector_path);
        ChessLensGame2 game2(config);
        
        std::cout << "ChessLens initialized. Starting game tracking...\n";
        std::cout << "Press Ctrl+C to stop.\n\n";
        
        // Main processing loop
        std::vector<double> frame_times;
        
        while (is_running) {
            auto t1 = std::chrono::high_resolution_clock::now();
            
            // Process frame from camera
            auto probs = game1.operate();
            
            // Update bindings
            game2.update_bindings();
            
            // Check if we should stop (no more frames)
            if (probs.empty()) {
                // Check if this is end of video/images or just a filtered frame
                if (game1.board_flag.get()) {
                    std::cout << "Board detection failed too many times. Stopping.\n";
                    break;
                }
                // Otherwise, just a filtered frame, continue
                continue;
            }
            
            // Process probabilities through context model
            game2.operate(probs);
            
            auto t2 = std::chrono::high_resolution_clock::now();
            
            double frame_time = std::chrono::duration<double>(t2 - t1).count();
            frame_times.push_back(frame_time);
            
            // Optional: Print progress every N frames
            if (frame_times.size() % 100 == 0) {
                std::cout << "Processed " << frame_times.size() << " frames...\r" << std::flush;
            }
        }
        
        std::cout << "\n\nProcessing complete. Finalizing...\n";
        
        // Calculate statistics
        if (frame_times.empty()) {
            std::cout << "No frames were processed.\n";
            return 0;
        }
        
        double avg_frame = calculate_average(frame_times);
        double total_time = std::accumulate(frame_times.begin(), frame_times.end(), 0.0);
        int frame_count = frame_times.size();
        
        // Final binding
        game2.bind();
        
        // Get game history
        auto fens = game2.get_history(true);
        
        // Print timing statistics
        std::cout << "\n=== Performance Statistics ===\n";
        std::cout << "Avg Image Loading:      " 
                  << (game1.avg_times.load * 1000.0 / frame_count) << " ms\n";
        std::cout << "Avg Board Detection:    " 
                  << (game1.avg_times.board_detection * 1000.0 / frame_count) << " ms\n";
        std::cout << "Avg Wakeup:             " 
                  << (game1.avg_times.wakeup * 1000.0 / frame_count) << " ms\n";
        std::cout << "Avg Occlusion:          " 
                  << (game1.avg_times.occlusion * 1000.0 / frame_count) << " ms\n";
        std::cout << "Avg Piece Recognition:  " 
                  << (game1.avg_times.piece_recognition * 1000.0 / frame_count) << " ms\n";
        std::cout << "Avg HMM:                " 
                  << (game2.avg_times.hmm * 1000.0 / frame_count) << " ms\n";
        std::cout << "\n";
        std::cout << "Avg Frame Time:         " << (avg_frame * 1000.0) << " ms\n";
        std::cout << "Frame Count:            " << frame_count << "\n";
        std::cout << "Total Time:             " << (total_time * 1000.0) << " ms\n";
        std::cout << "FPS:                    " << (frame_count / total_time) << "\n";
        
        // Process FENs (remove timestamps if present)
        std::vector<std::string> clean_fens;
        for (const auto& fen : fens) {
            size_t dash_pos = fen.find(" - ");
            if (dash_pos != std::string::npos) {
                clean_fens.push_back(fen.substr(0, dash_pos));
            } else {
                clean_fens.push_back(fen);
            }
        }
        
        // Generate PGN
        std::string pgn = ChessUtils::fens_to_pgn(clean_fens);
        
        // Write output to file
        std::string output_path = "Game/game_out.txt";
        std::ofstream outfile(output_path);
        if (outfile.is_open()) {
            // Write FENs
            for (const auto& fen : clean_fens) {
                outfile << fen << "\n";
            }
            outfile << "\n";
            
            // Write PGN
            outfile << pgn << "\n";
            
            outfile.close();
            std::cout << "\nGame saved to: " << output_path << "\n";
        } else {
            std::cerr << "Failed to open output file: " << output_path << "\n";
        }
        
        // Print PGN to console
        if (!pgn.empty()) {
            std::cout << "\n=== PGN ===\n";
            std::cout << pgn << "\n";
        } else {
            std::cout << "\nNo PGN generated (may need chess library integration)\n";
        }
        
        // Cleanup
        game1.quit();
        game2.quit();
        
        std::cout << "\nDone!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
#ifdef USE_HTTP
        curl_global_cleanup();
#endif
        return 1;
    }
    
#ifdef USE_HTTP
    // Cleanup curl
    curl_global_cleanup();
#endif
    
    return 0;
}