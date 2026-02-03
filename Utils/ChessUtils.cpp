#include "ChessUtils.h"
#include <algorithm>
#include <cctype>
#include <random>
#include <ctime>

namespace ChessUtils {

std::vector<float> fen_to_tensor(const std::string& fen) {
    std::vector<float> board(13 * 8 * 8, 0.0f);
    
    // Initialize empty channel (12) to all 1s
    for (int i = 0; i < 64; ++i) {
        board[12 * 64 + i] = 1.0f;
    }
    
    // Extract position part from FEN (before first space)
    std::string position = fen;
    size_t space_pos = fen.find(' ');
    if (space_pos != std::string::npos) {
        position = fen.substr(0, space_pos);
    }
    
    const std::string pieces = "PNBRQKpnbrqk";
    int rank = 0, file = 0;
    
    for (char c : position) {
        if (c == '/') {
            rank++;
            file = 0;
        } else if (std::isdigit(c)) {
            file += (c - '0');
        } else {
            size_t piece_idx = pieces.find(c);
            if (piece_idx != std::string::npos) {
                int pos = rank * 8 + file;
                board[12 * 64 + pos] = 0.0f;  // Clear empty channel
                board[piece_idx * 64 + pos] = 1.0f;
                file++;
            }
        }
    }
    
    return board;
}

std::string tensor_to_fen(std::vector<int>& board) {
    Utils::Matrix<int> board_m(board, {1,8,8});
    return tensor_to_fen(board_m);
}

std::string tensor_to_fen(Utils::Matrix<int>& board) {
    // board is [8, 8] flattened
    const std::string pieces = "PNBRQKpnbrqk1";
    std::string fen;
    
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            // Find max channel at this position
            int ch = board[{0, rank, file}];
            
            if (ch == 12) {
                empty++;
            } else {
                if (empty > 0) {
                    fen += std::to_string(empty);
                    empty = 0;
                }
                fen += pieces[ch];
            }
        }
        if (empty > 0) {
            fen += std::to_string(empty);
        }
        if (rank > 0) {
            fen += '/';
        }
    }
    
    return fen;
}

std::string tensor_to_fen_max(const std::vector<float>& probs) {
    // probs is [8, 8, 13] flattened
    const std::string pieces = "PNBRQKpnbrqk1";
    std::string fen;
    
    for (int rank = 0; rank < 8; ++rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            // Find argmax over 13 channels
            int max_ch = 0;
            float max_val = probs[rank * 8 * 13 + file * 13 + 0];
            
            for (int ch = 1; ch < 13; ++ch) {
                float val = probs[rank * 8 * 13 + file * 13 + ch];
                if (val > max_val) {
                    max_val = val;
                    max_ch = ch;
                }
            }
            
            if (max_ch == 12) {
                empty++;
            } else {
                if (empty > 0) {
                    fen += std::to_string(empty);
                    empty = 0;
                }
                fen += pieces[max_ch];
            }
        }
        if (empty > 0) {
            fen += std::to_string(empty);
        }
        if (rank < 7) {
            fen += '/';
        }
    }
    
    return fen;
}

cv::Mat fen_to_png(const std::string& fen, int width, int height) {
    // Placeholder implementation
    // To implement this properly, you would need:
    // 1. A chess piece rendering library (like python-chess with cairosvg)
    // 2. Or pre-rendered piece images and composite them
    // 3. Or use a chess GUI library
    
    // For now, return empty Mat
    // You can implement this using piece sprites and cv::Mat composition
    return cv::Mat();
}

std::string fens_to_pgn(const std::vector<std::string>& fens) {
    // Placeholder implementation
    // To implement this properly, you would need:
    // 1. A chess library (like Stockfish, python-chess equivalent in C++)
    // 2. Parse each FEN into a board position
    // 3. Find legal moves between consecutive positions
    // 4. Build PGN from move list
    
    // For now, return empty string
    // Consider using libraries like:
    // - Stockfish (complex but powerful)
    // - thc-chess-library
    // - Your own ChessGameState class with move generation
    return "";
}

std::vector<float> rand_one_hot(int seed) {
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        gen.seed(std::time(nullptr));
    }
    
    std::uniform_int_distribution<> dis(0, 12);
    
    std::vector<float> board(13 * 8 * 8, 0.0f);
    
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int piece = dis(gen);
            int pos = rank * 8 + file;
            board[piece * 64 + pos] = 1.0f;
        }
    }
    
    return board;
}

std::vector<int> rand_ints(int seed) {
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        gen.seed(std::time(nullptr));
    }
    
    std::uniform_int_distribution<> dis(0, 12);
    
    std::vector<int> board(64);
    
    for (int i = 0; i < 64; ++i) {
        board[i] = dis(gen);
    }
    
    return board;
}

std::string rand_fen(int seed) {
    auto board = rand_one_hot(seed);
    return tensor_to_fen_max(board);
}

} // namespace ChessUtils