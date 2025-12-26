// src/main.cpp (FINAL, DEFINITIVE - Unified Trainer)
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <sstream>
#include "lmdb++.h"
#include "utils.hpp"
#include "inference.hpp"
#include "hnswlib/hnswlib.h"

using NextGivenCurrentCounts = std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint64_t>>;
using PrevGivenCurrentCounts = std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint64_t>>;

void trainModel(const std::string& corpusPath, const std::string& dbPath) {
    const int VECTOR_DIMENSION = 256;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Starting FMM model training (V4.2 - Unified BPE Model)..." << std::endl;
    std::ifstream corpusFile(corpusPath);
    if (!corpusFile.is_open()) {
        std::cerr << "Error: Could not open corpus file at " << corpusPath << std::endl;
        return;
    }

    std::cout << "\n[Phase 1: Building Statistics from BPE Corpus]" << std::endl;
    NextGivenCurrentCounts p_next_given_current_counts;
    PrevGivenCurrentCounts p_prev_given_current_counts;
    std::string line;
    uint32_t max_id = 0;
    while (std::getline(corpusFile, line)) {
        std::stringstream ss(line);
        uint32_t id;
        std::vector<uint32_t> id_tokens;
        while(ss >> id) {
            id_tokens.push_back(id);
            if (id > max_id) max_id = id;
        }
        if (id_tokens.size() < 2) continue;
        for (size_t i = 0; i < id_tokens.size() - 1; ++i) {
            p_next_given_current_counts[id_tokens[i]][id_tokens[i+1]]++;
            p_prev_given_current_counts[id_tokens[i+1]][id_tokens[i]]++;
        }
    }
    std::cout << "Statistics built. Max token ID found: " << max_id << std::endl;

    try {
        std::string command = "mkdir -p " + dbPath;
        system(command.c_str());
        lmdb::env env = lmdb::env(dbPath.c_str(), MDB_WRITEMAP, 0664);

        { 
            lmdb::txn txn = lmdb::txn(env, nullptr, 0);
            lmdb::dbi p_next_dbi = lmdb::dbi(txn, "p_next_given_current", MDB_CREATE | MDB_INTEGERKEY);
            lmdb::dbi p_prev_dbi = lmdb::dbi(txn, "p_prev_given_current", MDB_CREATE | MDB_INTEGERKEY);

            std::cout << "Writing forward statistical distributions..." << std::endl;
            for (const auto& pair : p_next_given_current_counts) {
                uint64_t total_count = 0;
                for (const auto& next_pair : pair.second) total_count += next_pair.second;
                if (total_count == 0) continue;
                std::vector<ProbEntry> dist;
                dist.reserve(pair.second.size());
                for (const auto& next_pair : pair.second) {
                    dist.push_back({next_pair.first, static_cast<float>(next_pair.second) / total_count});
                }
                lmdb::put(txn, p_next_dbi, lmdb::val(pair.first), lmdb::val(dist));
            }
            std::cout << "Writing reverse statistical distributions..." << std::endl;
            for (const auto& pair : p_prev_given_current_counts) {
                uint64_t total_count = 0;
                for (const auto& prev_pair : pair.second) total_count += prev_pair.second;
                if (total_count == 0) continue;
                std::vector<ProbEntry> dist;
                dist.reserve(pair.second.size());
                for (const auto& prev_pair : pair.second) {
                    dist.push_back({prev_pair.first, static_cast<float>(prev_pair.second) / total_count});
                }
                lmdb::put(txn, p_prev_dbi, lmdb::val(pair.first), lmdb::val(dist));
            }
            std::cout << "Statistical tables written." << std::endl;
        }

        std::cout << "\n[Phase 2: Building Question-to-Answer Memory Bank]" << std::endl;
        hnswlib::L2Space space(VECTOR_DIMENSION);
        hnswlib::HierarchicalNSW<float>* ann_index = new hnswlib::HierarchicalNSW<float>(&space, 70000, 16, 200, 100, true);
        corpusFile.clear();
        corpusFile.seekg(0, std::ios::beg);
        uint64_t memory_idx = 0;
        std::vector<uint32_t> current_instruction_ids;
        const uint32_t INSTRUCTION_ID = 3;
        const uint32_t RESPONSE_ID = 4;

        {
            lmdb::txn mem_txn(env, nullptr, 0);
            lmdb::dbi mem_dbi = lmdb::dbi(mem_txn, "memory_outcomes", MDB_CREATE | MDB_INTEGERKEY);
            while(std::getline(corpusFile, line)) {
                std::vector<uint32_t> id_tokens;
                std::stringstream ss(line);
                uint32_t id;
                while(ss >> id) { id_tokens.push_back(id); }
                if (id_tokens.empty()) continue;
                if (id_tokens[0] == INSTRUCTION_ID) {
                    current_instruction_ids = std::vector<uint32_t>(id_tokens.begin() + 1, id_tokens.end());
                } else if (id_tokens[0] == RESPONSE_ID && !current_instruction_ids.empty() && id_tokens.size() > 1) {
                    std::vector<float> vec(VECTOR_DIMENSION, 0.0f);
                    for(const auto& token_id : current_instruction_ids) {
                        vec[token_id % VECTOR_DIMENSION] += 1.0f;
                    }
                    ann_index->addPoint(vec.data(), memory_idx);
                    uint32_t first_response_token_id = id_tokens[1];
                    lmdb::put(mem_txn, mem_dbi, lmdb::val(memory_idx), lmdb::val(first_response_token_id));
                    if (++memory_idx % 10000 == 0) {
                        std::cout << "Indexed " << memory_idx << " Q&A memories..." << std::endl;
                    }
                    current_instruction_ids.clear();
                }
            }
        }
        std::cout << "Saving ANN index to disk..." << std::endl;
        ann_index->saveIndex(dbPath + "/ann_index.bin");
        delete ann_index;
    } catch (const std::exception& e) { std::cerr << "Error during training: " << e.what() << std::endl; }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\nTraining complete in " << duration.count() << " seconds." << std::endl;
}
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: \n" << "  " << argv[0] << " train <path_to_corpus.txt> <path_to_db>\n" << "  " << argv[0] << " predict <path_to_db> <path_to_tokenizer.json>\n";
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "train") {
        trainModel(argv[2], argv[3]);
    } else if (mode == "predict") {
        InferenceEngine engine(argv[2], argv[3]);
        std::cout << "\n--- FMM Chatbot Initialized (Unified Model v4.2) ---" << std::endl;
        std::cout << "Enter your prompt. Type '[EXIT]' to quit." << std::endl;
        std::string prompt;
        while (true) {
            std::cout << "\n> ";
            std::getline(std::cin, prompt);
            if (prompt == "[EXIT]") { break; }
            std::string current_context = prompt;
            std::cout << ">> " << prompt;
            current_context += " [RESPONSE]";
            for (int i = 0; i < 80; ++i) {
                std::string prediction = engine.predict_next_token(current_context);
                if (prediction == "[STOP]" || prediction.find('[') != std::string::npos || prediction.empty()) {
                    break;
                }
                std::cout << prediction << std::flush;
                current_context += prediction;
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'." << std::endl;
        return 1;
    }
    return 0;
}