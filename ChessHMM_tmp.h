#ifndef CHESSHMM
#define CHESSHMM

#include "ChessGameState.h"
#include <vector>
#include <queue>
#include <set>
#include <string>

using namespace std;

class HMMState {
    friend class ChessHMM;

public:
    HMMState(ChessGameState* position);
    HMMState(HMMState& parent, ChessGameState* position, bool is_selfloop = false);

    double eval_prob(const vector<float>& obs_probs); // 8x8x13
    vector<HMMState*> get_children();

    bool operator<(const HMMState& other) const;
    bool operator==(const HMMState& other) const;

private:
    ChessGameState* game_state;
    HMMState* parent;
    vector<HMMState*> children;

    int timestep;
    double prob;

    bool is_self_loop;
    bool is_children_computed;

    double observation_prob;

    double transition_prob();
    double parent_prob();
    double get_child_transition_prob() const;

    void compute_children();
};

class HMMStateFactory {
public:
    static HMMState* create_state(HMMState* parent, ChessGameState* position, bool is_selfloop = false);

private:
    struct Registry {
        vector<HMMState*> states;
        ~Registry();
    };

    static Registry registry;
};

class ChessHMM {
public:
    ChessHMM(int max_width, string fen = STARTING_POSITION_FEN);
    ~ChessHMM();

    int top_t() const;
    int top_bind_t() const;

    void set_probs(int timestep, const vector<float>& obs_probs);
    void bind(int timestep);

    string print(int timestep) const;

    vector<int> get_history(bool include_non_bound = false) const;
    string get_pgn() const;

private:
    struct CompareProbs {
        bool operator()(const HMMState* a, const HMMState* b) const;
    };

    HMMState* root;
    vector<multiset<HMMState*, CompareProbs>*> timed_tree_states;

    int _top_bind_t;
    size_t max_width;
};

#endif
