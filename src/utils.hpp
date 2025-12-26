// src/utils.hpp (With a better tokenizer)

#ifndef FMM_UTILS_HPP
#define FMM_UTILS_HPP

#include <string>
#include <vector>
#include <sstream>
#include <algorithm> // For std::transform
#include <cctype>    // For std::tolower

inline std::string sanitize_token(const std::string& s) {
    std::string sanitized;
    // Remove punctuation from the end
    for (char c : s) {
        if (!ispunct(c)) {
            sanitized += std::tolower(c);
        }
    }
    return sanitized;
}

inline std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (tokenStream >> token) { // Use >> to automatically handle spaces
        std::string sanitized = sanitize_token(token);
        if (!sanitized.empty()) {
            tokens.push_back(sanitized);
        }
    }
    return tokens;
}

struct ProbEntry {
    uint32_t token_id;
    float probability;
};

#endif // FMM_UTILS_HPP