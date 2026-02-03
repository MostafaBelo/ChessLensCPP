#include "HMM.h"
#include <iostream>

namespace ContextAwareModels {

HMM::HMM(int breadth, double delay, int bind_period)
    : breadth_(breadth), delay_(delay), bind_period_(bind_period) {
    
    // Create underlying ChessHMM with starting position
    model_ = new ChessHMM(breadth, STARTING_POSITION_FEN);
}

HMM::~HMM() {
    delete model_;
}

int HMM::top_t() const {
    return model_->top_t();
}

int HMM::top_bind_t() const {
    return model_->top_bind_t();
}

void HMM::bind() {
    model_->bind(model_->top_t());
}

void HMM::set_probs(int timestep, 
                    const std::vector<float>& piece_matrix,
                    const std::chrono::steady_clock::time_point& actual_frame_time) {
    
    // Store timestamp for this frame
    timestamp_map_[timestep] = actual_frame_time;
    
    // Set probabilities in underlying model
    model_->set_probs(timestep, piece_matrix);
    
    // Original Python code had commented out auto-binding logic:
    // if (((model_->top_t() % bind_period_) == 0) && 
    //     (model_->top_t() >= bind_period_ + delay_) && 
    //     (model_->top_t() - delay_ - bind_period_ + 1 > model_->top_bind_t())) {
    //     model_->bind(model_->top_t() - delay_ + 1);
    // }
}

bool HMM::check_bind(const std::chrono::steady_clock::time_point& current_time) {
    int bind_at = -1;
    
    // Find the latest timestep that has exceeded the delay threshold
    for (int i = model_->top_bind_t() + 1; i < model_->top_t(); ++i) {
        auto it = timestamp_map_.find(i);
        if (it != timestamp_map_.end()) {
            auto elapsed = std::chrono::duration<double>(current_time - it->second).count();
            if (elapsed >= delay_) {
                bind_at = i;
            }
        }
    }
    
    if (bind_at == -1) {
        return false;
    }

    cout << "binding at " << bind_at << "\n";
    
    try {
        model_->bind(bind_at);
        return true;
    } catch (...) {
        // Binding failed
        return false;
    }
}

std::string HMM::print(int timestep) const {
    return model_->print(timestep);
}

Utils::Matrix<int> HMM::get_history(bool include_non_bound) const {
    return model_->get_history(include_non_bound);
}

std::string HMM::get_pgn() const {
    return model_->get_pgn();
}

} // namespace ContextAwareModels