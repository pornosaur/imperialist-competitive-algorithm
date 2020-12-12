//
// Created by Patrik Patera.
//

#ifndef IMPERIALIST_COMPETITIVE_ALGORITHM_UTILS_H
#define IMPERIALIST_COMPETITIVE_ALGORITHM_UTILS_H

#include <cstddef>
#include <limits>

constexpr double DBL_MAX = std::numeric_limits<double>::max();

constexpr double DBL_MIN = std::numeric_limits<double>::min();

constexpr size_t SIZET_MAX = std::numeric_limits<size_t>::max();

enum RESULT_FLAGS {
    OK,
    FALSE,
    INVALID_ARGS,
    FAIL,
};

#endif //IMPERIALIST_COMPETITIVE_ALGORITHM_UTILS_H
