#pragma once

#include "../ChessHMM.h"
#include <map>
#include <chrono>
#include <vector>
#include <string>

namespace ContextAwareModels {

/**
 * Wrapper around ChessHMM that adds time-delayed binding logic
 * Manages timestamps and automatic binding based on real-time delays
 */
class HMM {
public:
    /**
     * Construct HMM wrapper
     * 
     * @param breadth Maximum width of search tree
     * @param delay Time delay (in seconds) before binding positions
     * @param bind_period How often to check for binding (not currently used)
     */
    HMM(int breadth = 30, double delay = 120.0, int bind_period = 20);
    
    /**
     * Destructor
     */
    ~HMM();
    
    /**
     * Get the current top timestep
     * @return Maximum timestep with probabilities set
     */
    int top_t() const;
    
    /**
     * Get the current top bound timestep
     * @return Maximum timestep that has been bound
     */
    int top_bind_t() const;
    
    /**
     * Bind position at current top timestep
     */
    void bind();
    
    /**
     * Set observation probabilities for a timestep
     * 
     * @param timestep Timestep index
     * @param piece_matrix Negative log probabilities [8, 8, 13] flattened
     * @param actual_frame_time Real-world timestamp for this frame
     */
    void set_probs(int timestep, 
                   const std::vector<float>& piece_matrix, 
                   const std::chrono::steady_clock::time_point& actual_frame_time);
    
    /**
     * Check if sufficient time has passed and bind if ready
     * 
     * @param current_time Current real-world time
     * @return true if binding occurred, false otherwise
     */
    bool check_bind(const std::chrono::steady_clock::time_point& current_time);
    
    /**
     * Print HMM state at a given timestep
     * 
     * @param timestep Timestep to print
     * @return String representation
     */
    std::string print(int timestep) const;
    
    /**
     * Get history of board positions
     * 
     * @param include_non_bound Include unbound positions
     * @return Vector of board state indices
     */
    Utils::Matrix<int> get_history(bool include_non_bound = false) const;
    
    /**
     * Get PGN string of the game
     * @return PGN string
     */
    std::string get_pgn() const;
    
    /**
     * Access to underlying model (for advanced usage)
     */
    ChessHMM* get_model() { return model_; }
    const ChessHMM* get_model() const { return model_; }

private:
    ChessHMM* model_;
    
    int breadth_;
    int bind_period_;
    double delay_;
    
    // Map timestep to real-world time
    std::map<int, std::chrono::steady_clock::time_point> timestamp_map_;
};

} // namespace ContextAwareModels