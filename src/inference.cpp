// src/inference.cpp (FINAL, DEFINITIVE - Corrected for C++ Tokenizer)

#include "inference.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <sstream>

#include "utils.hpp"
#include "nlohmann/json.hpp"

// Helper function to find a probability within a deserialized distribution
float get_prob(const std::vector<ProbEntry>& dist, uint32_t target_id) {
    for (const auto& entry : dist) if (entry.token_id == target_id) return entry.probability;
    return 0.0f;
}

// THE FIX: The constructor no longer needs the tokenizer path.
InferenceEngine::InferenceEngine(const std::string& dbPath)
    : env(dbPath.c_str(), MDB_RDONLY, 0), space(256)
{
    std::cout << "Initializing Inference Engine..." << std::endl;
    try {
        // We go back to loading the simple vocab from the DB, which is correct for our word-based trainer.
        load_vocabulary_from_db();
        
        std::string index_path = dbPath + "/ann_index.bin";
        std::cout << "Loading ANN index from " << index_path << std::endl;
        ann_index = new hnswlib::HierarchicalNSW<float>(&space, index_path);
        if (ann_index->getCurrentElementCount() == 0) {
            std::cerr << "Warning: ANN index is empty or could not be loaded." << std::endl;
        }
        std::cout << "ANN index loaded with " << ann_index->getCurrentElementCount() << " vectors." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error during initialization: " << e.what() << std::endl;
        exit(1);
    }
    std::cout << "Inference Engine ready." << std::endl;
}

InferenceEngine::~InferenceEngine() {
    if (ann_index) delete ann_index;
}

// THE FIX: Renamed this function back to its original, correct purpose.
void InferenceEngine::load_vocabulary_from_db() {
    std::cout << "Loading vocabulary from database..." << std::endl;
    lmdb::txn txn = lmdb::txn(env, nullptr, MDB_RDONLY);
    lmdb::dbi vocab_dbi = lmdb::dbi(txn, "vocab_to_id", 0);
    MDB_cursor* cursor;
    if (auto rc = mdb_cursor_open(txn, vocab_dbi, &cursor)) {
        throw lmdb::exception("mdb_cursor_open", rc);
    }
    MDB_val key, data;
    while (mdb_cursor_get(cursor, &key, &data, MDB_NEXT) == 0) {
        std::string token_str(static_cast<char*>(key.mv_data), key.mv_size);
        uint32_t token_id = *static_cast<uint32_t*>(data.mv_data);
        vocab_to_id[token_str] = token_id;
    }
    mdb_cursor_close(cursor);
    id_to_vocab.resize(vocab_to_id.size() + 100); // Add a buffer
    for(const auto& pair : vocab_to_id) {
        if(pair.second >= id_to_vocab.size()) id_to_vocab.resize(pair.second + 100);
        id_to_vocab[pair.second] = pair.first;
    }
    std::cout << "Vocabulary loaded. Total tokens: " << vocab_to_id.size() << std::endl;
}

std::string InferenceEngine::predict_next_token(const std::string& context) {
    const float ATTENTION_MULTIPLIER = 10000.0f;
    const float REPETITION_PENALTY = 1.5f;
    const int TOP_K = 40;
    const int NUM_NEIGHBORS = 25;
    const int VECTOR_DIMENSION = 256;

    // THE FIX: Use our own robust tokenizer.
    std::vector<std::string> context_tokens = tokenize(context);
    if (context_tokens.empty()) return "[EMPTY_CONTEXT]";

    bool is_responding_turn = (context_tokens.back() == "[RESPONSE]");

    try {
        if (is_responding_turn) {
            // --- MODE 1: RESPONDING (Pure Retrieval from Q&A Memory) ---
            std::vector<float> memory_scores(id_to_vocab.size(), 0.0f);
            std::vector<float> query_vec(VECTOR_DIMENSION, 0.0f);
            for(size_t i = 0; i < context_tokens.size() - 1; ++i) { // Exclude [RESPONSE]
                if(vocab_to_id.count(context_tokens[i])) {
                    query_vec[vocab_to_id.at(context_tokens[i]) % VECTOR_DIMENSION] += 1.0f;
                }
            }

            if (ann_index->getCurrentElementCount() > 0) {
                auto result = ann_index->searchKnn(query_vec.data(), NUM_NEIGHBORS);
                lmdb::txn txn(env, nullptr, MDB_RDONLY);
                lmdb::dbi mem_dbi = lmdb::dbi(txn, "memory_outcomes", MDB_INTEGERKEY);
                while(!result.empty()) {
                    MDB_val outcome_data;
                    uint64_t mem_idx = result.top().second;
                    lmdb::val mem_key(mem_idx);
                    if (mdb_get(txn, mem_dbi, &mem_key.mdb_val, &outcome_data) == 0) {
                        uint32_t outcome_id = *static_cast<uint32_t*>(outcome_data.mv_data);
                        memory_scores[outcome_id] += 1.0f / (1.0f + result.top().first);
                    }
                    result.pop();
                }
            }
            
            uint32_t best_token_id = 0;
            float max_score = -1.0f;
            for (uint32_t i = 0; i < memory_scores.size(); ++i) {
                if (memory_scores[i] > max_score) { max_score = memory_scores[i]; best_token_id = i; }
            }
            if (max_score <= 0.0f) return "[NO_MEMORY_MATCH]";
            return id_to_vocab[best_token_id];

        } else {
            // --- MODE 2: CONTINUING (Creative Autocomplete with Attention) ---
            std::vector<float> final_scores(id_to_vocab.size(), 0.0f);
            lmdb::txn txn(env, nullptr, MDB_RDONLY);
            lmdb::dbi p_next_dbi = lmdb::dbi(txn, "p_next_given_current", MDB_INTEGERKEY);
            lmdb::dbi p_prev_dbi = lmdb::dbi(txn, "p_prev_given_current", MDB_INTEGERKEY);
            MDB_val db_data;
            
            std::vector<uint32_t> context_ids;
            for(const auto& token_str : context_tokens) if(vocab_to_id.count(token_str)) context_ids.push_back(vocab_to_id.at(token_str));
            if(context_ids.empty()) return "[UNKNOWN_CONTEXT]";

            uint32_t last_token_id = context_ids.back();
            lmdb::val last_token_key(last_token_id);
            
            if (mdb_get(txn, p_next_dbi, &last_token_key.mdb_val, &db_data) == 0) {
                const ProbEntry* entries = static_cast<const ProbEntry*>(db_data.mv_data);
                for (size_t i = 0; i < db_data.mv_size / sizeof(ProbEntry); ++i) final_scores[entries[i].token_id] += entries[i].probability;
            }

            if (context_ids.size() > 1) {
                for (size_t i = 0; i < context_ids.size() - 1; ++i) {
                    uint32_t prev_token_id = context_ids[i];
                    lmdb::val prev_token_key(prev_token_id);
                    float p_last_given_prev = 0.0f, p_prev_given_last = 0.0f;
                    if (mdb_get(txn, p_next_dbi, &prev_token_key.mdb_val, &db_data) == 0) {
                        std::vector<ProbEntry> dist(static_cast<const ProbEntry*>(db_data.mv_data), static_cast<const ProbEntry*>(db_data.mv_data) + db_data.mv_size / sizeof(ProbEntry));
                        p_last_given_prev = get_prob(dist, last_token_id);
                    }
                    if (mdb_get(txn, p_prev_dbi, &last_token_key.mdb_val, &db_data) == 0) {
                        std::vector<ProbEntry> dist(static_cast<const ProbEntry*>(db_data.mv_data), static_cast<const ProbEntry*>(db_data.mv_data) + db_data.mv_size / sizeof(ProbEntry));
                        p_prev_given_last = get_prob(dist, prev_token_id);
                    }
                    float attention_score = p_last_given_prev * p_prev_given_last;
                    if (attention_score < 1e-9) continue;
                    if (mdb_get(txn, p_next_dbi, &prev_token_key.mdb_val, &db_data) == 0) {
                        const ProbEntry* entries = static_cast<const ProbEntry*>(db_data.mv_data);
                        for (size_t j = 0; j < db_data.mv_size / sizeof(ProbEntry); ++j) {
                            final_scores[entries[j].token_id] += ATTENTION_MULTIPLIER * attention_score * entries[j].probability;
                        }
                    }
                }
            }
            
            size_t lookback = std::min((size_t)15, context_ids.size());
            for (size_t i = 0; i < lookback; ++i) {
                final_scores[context_ids[context_ids.size() - 1 - i]] /= REPETITION_PENALTY;
            }
            
            std::vector<std::pair<float, uint32_t>> sorted_scores;
            for (uint32_t i = 0; i < final_scores.size(); ++i) {
                if (final_scores[i] > 1e-9) { sorted_scores.push_back({final_scores[i], i}); }
            }
            std::sort(sorted_scores.rbegin(), sorted_scores.rend());
            
            if (sorted_scores.empty()) return "[NO_VALID_PREDICTION]";
            if (sorted_scores.size() > TOP_K) { sorted_scores.resize(TOP_K); }

            double total_score = 0.0;
            for (const auto& pair : sorted_scores) { total_score += pair.first; }
            if (total_score < 1e-9) return "[NO_CONFIDENT_PREDICTION]";
            
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<> dis(0.0, total_score);
            
            uint32_t best_token_id = sorted_scores[0].second;
            double cumulative_score = 0.0;
            double sample = dis(gen);
            for (const auto& pair : sorted_scores) {
                cumulative_score += pair.first;
                if (sample < cumulative_score) {
                    best_token_id = pair.second;
                    break;
                }
            }
            return id_to_vocab[best_token_id];
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during prediction: " << e.what() << std::endl;
        return "[DB_ERROR]";
    }
}