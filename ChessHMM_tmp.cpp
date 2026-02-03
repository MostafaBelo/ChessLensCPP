#include "ChessHMM.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <unordered_set>

using namespace std;

static inline float OBS(const vector<float>& obs, int i, int j, int k) {
    return obs[(i * 8 + j) * 13 + k];
}

// --------------------------------------------------
// HMMState
// --------------------------------------------------

HMMState::HMMState(ChessGameState* position)
    : game_state(position),
      parent(nullptr),
      timestep(0),
      prob(0),
      observation_prob(0),
      is_self_loop(false),
      is_children_computed(false) {}

HMMState::HMMState(HMMState& parent, ChessGameState* position, bool is_selfloop)
    : game_state(position),
      parent(&parent),
      timestep(parent.timestep + 1),
      prob(0),
      observation_prob(0),
      is_self_loop(is_selfloop),
      is_children_computed(false) {}

double HMMState::eval_prob(const vector<float>& obs_probs) {
    if (obs_probs.size() != 8 * 8 * 13)
        throw invalid_argument("obs_probs must be 8x8x13");

    observation_prob = 0.0;

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            observation_prob += OBS(obs_probs, i, j, game_state->current_position[i][j]);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int best = 0;
            for (int k = 1; k < 13; ++k)
                if (OBS(obs_probs, i, j, k) < OBS(obs_probs, i, j, best))
                    best = k;

            if (game_state->current_position[i][j] != parent->game_state->current_position[i][j] ||
                game_state->current_position[i][j] != best) {
                observation_prob += OBS(obs_probs, i, j, game_state->current_position[i][j]);
            }
        }
    }

    prob = observation_prob + transition_prob() + parent_prob();
    return prob;
}

vector<HMMState*> HMMState::get_children() {
    if (!is_children_computed)
        compute_children();
    return children;
}

bool HMMState::operator<(const HMMState& other) const {
    return prob < other.prob;
}

bool HMMState::operator==(const HMMState& other) const {
    return prob == other.prob;
}

double HMMState::transition_prob() {
    double p = parent->get_child_transition_prob();
    if (!is_self_loop)
        p += 20;
    return p;
}

double HMMState::parent_prob() {
    return parent->prob;
}

double HMMState::get_child_transition_prob() const {
    if (!is_children_computed)
        throw logic_error("Children not computed");
    return log(children.size());
}

void HMMState::compute_children() {
    children.push_back(HMMStateFactory::create_state(this, game_state, true));
    for (ChessGameState* pos : game_state->get_children())
        children.push_back(HMMStateFactory::create_state(this, pos));
    is_children_computed = true;
}

// --------------------------------------------------
// Factory
// --------------------------------------------------

HMMStateFactory::Registry HMMStateFactory::registry;

HMMState* HMMStateFactory::create_state(HMMState* parent, ChessGameState* position, bool is_selfloop) {
    HMMState* s = parent
        ? new HMMState(*parent, position, is_selfloop)
        : new HMMState(position);
    registry.states.push_back(s);
    return s;
}

HMMStateFactory::Registry::~Registry() {
    for (auto s : states)
        delete s;
}

// --------------------------------------------------
// ChessHMM
// --------------------------------------------------

bool ChessHMM::CompareProbs::operator()(const HMMState* a, const HMMState* b) const {
    return a->prob < b->prob;
}

ChessHMM::ChessHMM(int max_width, string fen)
    : _top_bind_t(0),
      max_width(max_width) {
    root = HMMStateFactory::create_state(nullptr, GameStateFactory::create_state(fen));
    timed_tree_states.push_back(new multiset<HMMState*, CompareProbs>());
    timed_tree_states[0]->insert(root);
}

ChessHMM::~ChessHMM() {
    for (auto ptr : timed_tree_states)
        delete ptr;
}

int ChessHMM::top_t() const {
    return timed_tree_states.size() - 1;
}

int ChessHMM::top_bind_t() const {
    return _top_bind_t;
}

void ChessHMM::set_probs(int timestep, const vector<float>& obs_probs) {
    if ((timestep != top_t() && timestep != top_t() + 1) || timestep <= 0)
        throw invalid_argument("Invalid timestep");

    if (timestep == top_t())
        timed_tree_states[timestep]->clear();
    else
        timed_tree_states.push_back(new multiset<HMMState*, CompareProbs>());

    for (HMMState* s : *timed_tree_states[timestep - 1]) {
        for (HMMState* c : s->get_children()) {
            c->eval_prob(obs_probs);
            timed_tree_states[timestep]->insert(c);
        }
    }

    if (timed_tree_states[timestep]->size() > max_width) {
        auto it = timed_tree_states[timestep]->begin();
        advance(it, max_width);
        timed_tree_states[timestep]->erase(it, timed_tree_states[timestep]->end());
    }
}

void ChessHMM::bind(int timestep) {
    if (timestep > top_t() || timestep <= _top_bind_t)
        throw invalid_argument("Invalid timestep");

    HMMState* best = *timed_tree_states[top_t()]->begin();
    for (auto s = best; s; s = s->parent) {
        if (s->timestep <= timestep) {
            timed_tree_states[s->timestep]->clear();
            timed_tree_states[s->timestep]->insert(s);
        }
    }

    for (int t = timestep + 1; t <= top_t(); ++t) {
        unordered_set<HMMState*> valid;
        for (auto s : *timed_tree_states[t - 1])
            valid.insert(s);

        multiset<HMMState*, CompareProbs> filtered;
        for (auto s : *timed_tree_states[t])
            if (valid.count(s->parent))
                filtered.insert(s);

        timed_tree_states[t]->swap(filtered);
    }

    _top_bind_t = timestep;
}

string ChessHMM::print(int timestep) const {
    if (timestep < 0 || timestep > top_t())
        throw invalid_argument("Invalid timestep");

    string out;
    for (auto s : *timed_tree_states[timestep]) {
        out += "Prob: " + to_string(s->prob) + "\n";
        out += s->game_state->str() + "\n\n";
    }
    return out;
}

vector<int> ChessHMM::get_history(bool include_non_bound) const {
    int T = include_non_bound ? top_t() + 1 : top_bind_t() + 1;
    vector<int> hist(T * 8 * 8, 0);

    HMMState* s = *timed_tree_states[top_t()]->begin();
    for (; s; s = s->parent) {
        if (!include_non_bound && s->timestep > top_bind_t())
            continue;

        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                hist[(s->timestep * 64) + i * 8 + j] =
                    static_cast<int>(s->game_state->current_position[i][j]);
    }
    return hist;
}

string ChessHMM::get_pgn() const {
    return "";
}
