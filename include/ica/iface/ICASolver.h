//
// Created by Patrik Patera.
//

#ifndef IMPERIALIST_COMPETITIVE_ALGORITHM_ICASOLVER_H
#define IMPERIALIST_COMPETITIVE_ALGORITHM_ICASOLVER_H

#include <libconfig.h++>

#include <ica/iface/utils.h>
#include <functional>
#include <memory>
#include <iostream>
#include <string>

namespace solver {

    using objective_function = std::function<double(double *, size_t)>;

    struct ICASolver_t {
        const size_t dimension;

        const double *lower_bound;
        const double *upper_bound;
        double *const solution;

        const objective_function solving_function;

        const double tolerance;
        const size_t population_size;
        const size_t max_generations;

        ~ICASolver_t() {
            delete[] lower_bound;
            delete[] upper_bound;
            delete[] solution;
        }

    };

    const ICASolver_t default_solver = {0, nullptr, nullptr, nullptr, nullptr, DBL_MIN, 0, 0};

    struct ICAFunctions {
        const std::string name;
        const double global_min[2];
        const double solution;
        const objective_function solving_function;
    };

    struct ICAArgs {
        const size_t dimension;
        const size_t population_size;
        const size_t max_generations;
        const double tolerance;

        const bool methods[3]; // 0 - Serial version; 1 - SMP version; 2 - OpenCL version.
    };

    constexpr static double DEFAULT_TOLERANCE = DBL_MIN;

    constexpr static size_t DEFAULT_DIMENSION = 3;

    constexpr static size_t DEFAULT_POPULATION_SIZE = 100;

    constexpr static size_t DEFAULT_MAX_GENERATIONS = SIZET_MAX;

    const ICAArgs default_args = {DEFAULT_DIMENSION, DEFAULT_POPULATION_SIZE, SIZET_MAX, DEFAULT_TOLERANCE,
                                  {true, true, false}};

    ICAArgs load_config(const std::string &config_path);

}

#endif //IMPERIALIST_COMPETITIVE_ALGORITHM_ICASOLVER_H
