#pragma once

#include <vector>
#include <functional>
#include <string>
#include <cmath>

namespace Utils {

/**
 * Observable pattern implementation
 * Allows subscription to value changes with automatic notification
 * 
 * Example usage:
 *   Observable<bool> flag(false);
 *   flag.subscribe([](bool val) { std::cout << "Changed to: " << val << "\n"; });
 *   flag.set(true);  // Triggers notification
 */
template<typename T>
class Observable {
public:
    /**
     * Construct observable with initial value
     * @param initial_value Starting value
     */
    explicit Observable(const T& initial_value) : data_(initial_value) {}
    
    /**
     * Set new value and notify listeners if changed
     * @param value New value to set
     */
    void set(const T& value) {
        T old = data_;
        data_ = value;
        if (old != value) {
            notify();
        }
    }
    
    /**
     * Get current value
     * @return Current value
     */
    T get() const { 
        return data_; 
    }
    
    /**
     * Subscribe to value changes
     * @param listener Callback function called when value changes
     */
    void subscribe(std::function<void(const T&)> listener) {
        listeners_.push_back(listener);
    }
    
    /**
     * Clear all listeners
     */
    void clear_listeners() {
        listeners_.clear();
    }

private:
    T data_;
    std::vector<std::function<void(const T&)>> listeners_;
    
    /**
     * Notify all listeners of value change
     */
    void notify() {
        for (auto& listener : listeners_) {
            listener(data_);
        }
    }
};

struct Index3D {
    ssize_t i,j,k;
};

template <typename T>
class Matrix {
public:
    Matrix(Index3D shape) {
        Matrix<T>::shape = shape;
        data_ = std::vector<T>(shape.i * shape.j * shape.k);
    }
    Matrix(const std::vector<T>& data, Index3D shape) {
        data_ = data;
        Matrix<T>::shape = shape;
    }
    T& operator[](Index3D index) {
        return at(index);
    }
    std::string str() {
        std::string res = "[";
        for (int i = 0; i < 8; i++)
        {
            res += "[";
            for (int j = 0; j < 8; j++)
            {
                res += "[";
                for (int k = 0; k < 13; k++)
                {
                    res += to_string(at({i,j,k}));
                    if (k != 12) res += ", ";
                }
                res += "]";
                if (j != 7) res += "\n";
            }
            res += "]";
            if (i != 7) res += "\n\n";
        }
        res += "]";
        return res;
    }

    std::vector<T> data_;
    Index3D shape;
private:
    T& at(Index3D index) {
        ssize_t flat_index = index.i * (shape.j * shape.k) + index.j * shape.k + index.k;
        return data_[flat_index];
    }
};

} // namespace Utils