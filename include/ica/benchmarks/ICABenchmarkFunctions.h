//
// Created by Patrik Patera.
//

#ifndef IMPERIALIST_COMPETITIVE_ALGORITHM_ICABENCHMARKFUNCTIONS_H
#define IMPERIALIST_COMPETITIVE_ALGORITHM_ICABENCHMARKFUNCTIONS_H

#include <ica/iface/ICASolver.h>


namespace benchmark {

    class BenchmarkFunc {
    private:
        static double rosenbrock_func(const double *solution, size_t dim);

        static double griewank_func(const double *solution, size_t dim);

        static double schwefel_func(const double *solution, size_t dim);

        static std::vector<solver::ICAFunctions> benchmark_functions_list;

        static constexpr auto UNKNOW_FUNCTION = "unknown";

    public:
        BenchmarkFunc() = delete;

        static solver::ICASolver_t create_solver(const solver::ICAArgs &args, size_t id);

        static const std::string &get_func_name(size_t id);

        static size_t TOTAL_FUNCTIONS;
    };

}

#endif //IMPERIALIST_COMPETITIVE_ALGORITHM_ICABENCHMARKFUNCTIONS_H
