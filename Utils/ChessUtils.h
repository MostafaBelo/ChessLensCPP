#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Utils.h"

namespace ChessUtils {

/**
 * Convert FEN string to tensor representation [13, 8, 8]
 * Channels: P, N, B, R, Q, K, p, n, b, r, q, k, empty
 * 
 * @param fen FEN string (can include full FEN or just position)
 * @return Flattened vector representing [13, 8, 8] board
 */
std::vector<float> fen_to_tensor(const std::string& fen);

/**
 * Convert tensor [13, 8, 8] to FEN string
 * Uses argmax to determine piece at each square
 * 
 * @param board Flattened vector representing [13, 8, 8]
 * @return FEN string (position only, no move counters)
 */
std::string tensor_to_fen(std::vector<int>& board);
std::string tensor_to_fen(Utils::Matrix<int>& board);

/**
 * Convert probability tensor [8, 8, 13] to FEN string
 * Takes argmax over the last dimension
 * 
 * @param probs Flattened vector representing [8, 8, 13] probabilities
 * @return FEN string (position only)
 */
std::string tensor_to_fen_max(const std::vector<float>& probs);

/**
 * Generate PNG image from FEN position
 * Requires chess rendering library (placeholder implementation)
 * 
 * @param fen FEN string
 * @param width Output image width (default 200)
 * @param height Output image height (default 200)
 * @return PNG image as cv::Mat, or empty Mat if not implemented
 */
cv::Mat fen_to_png(const std::string& fen, int width = 200, int height = 200);

/**
 * Convert list of FENs to PGN format
 * Finds legal moves between consecutive positions
 * 
 * @param fens Vector of FEN strings
 * @return PGN string, or empty if conversion fails
 */
std::string fens_to_pgn(const std::vector<std::string>& fens);

/**
 * Generate random one-hot encoded board [13, 8, 8]
 * Useful for testing
 * 
 * @param seed Random seed (-1 for random seed)
 * @return Random board tensor
 */
std::vector<float> rand_one_hot(int seed = -1);

/**
 * Generate random integer board [8, 8]
 * Each value is 0-12 representing piece type
 * 
 * @param seed Random seed (-1 for random seed)
 * @return Random integer board
 */
std::vector<int> rand_ints(int seed = -1);

/**
 * Generate random FEN string
 * 
 * @param seed Random seed (-1 for random seed)
 * @return Random FEN string
 */
std::string rand_fen(int seed = -1);

} // namespace ChessUtils