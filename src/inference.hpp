// src/inference.hpp (FINAL, DEFINITIVE - Corrected Constructor)

#ifndef FMM_INFERENCE_HPP
#define FMM_INFERENCE_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "lmdb++.h"
#include "hnswlib/hnswlib.h"

class InferenceEngine {
private:
    lmdb::env env;
    std::unordered_map<std::string, uint32_t> vocab_to_id;
    std::vector<std::string> id_to_vocab;

    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<float>* ann_index = nullptr;

    // This function now needs the tokenizer path
    void load_vocabulary_from_tokenizer(const std::string& tokenizerPath);

public:
    // THE DEFINITIVE FIX:
    // The constructor now correctly takes two arguments, matching the call in main.cpp
    // and the definition in inference.cpp.
    InferenceEngine(const std::string& dbPath, const std::string& tokenizerPath);
    
    ~InferenceEngine();
    std::string predict_next_token(const std::string& context);
};

#endif // FMM_INFERENCE_HPP