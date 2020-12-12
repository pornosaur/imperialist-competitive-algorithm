//
// Created by Patrik Patera.
//
#include <cmath>
#include <ica/benchmarks/ICABenchmarkFunctions.h>

using namespace benchmark;

std::vector<solver::ICAFunctions> BenchmarkFunc::benchmark_functions_list
        = {{"Rosenbrock", {-5,   10},  1,        &BenchmarkFunc::rosenbrock_func},
           {"Griewank",   {-600, 600}, 1,        &BenchmarkFunc::griewank_func},
           {"Schwefel",   {-500, 500}, 420.9687, &BenchmarkFunc::schwefel_func}};

size_t BenchmarkFunc::TOTAL_FUNCTIONS = benchmark_functions_list.size();

const std::string & BenchmarkFunc::get_func_name(size_t id) {
    if (id >= TOTAL_FUNCTIONS) return UNKNOW_FUNCTION;

    return benchmark_functions_list[id].name;
}

double BenchmarkFunc::rosenbrock_func(const double *solution, size_t dim) {
    double sum = 0.;
    for (size_t i = 0; i < dim - 1; ++i) {
        sum += 100 * std::pow(-solution[i + 1] + solution[i] * solution[i], 2) + std::pow(-1 + solution[i], 2);
    }

    return std::abs(sum);
}

double BenchmarkFunc::griewank_func(const double *solution, size_t dim) {
    double sum = 0., mul = 1;
    for (size_t i = 0; i < dim; ++i) {
        sum += (solution[i] * solution[i]) / 4000;
        mul *= std::cos(solution[i] / std::sqrt(i + 1));
    }

    return std::abs(1 + sum - mul);
}

double BenchmarkFunc::schwefel_func(const double *solution, size_t dim) {
    double sum = 0.;
    for (size_t i = 0; i < dim; ++i) {
        sum += solution[i] * std::sin(std::sqrt(std::abs(solution[i])));
    }

    return 418.9829 * dim - sum;
}

solver::ICASolver_t BenchmarkFunc::create_solver(const solver::ICAArgs &args, size_t id) {
    if (id >= benchmark_functions_list.size()) return solver::default_solver;

    double *lower = new double[args.dimension];
    double *upper = new double[args.dimension];
    double *solution = new double[args.dimension];

    for (size_t i = 0; i < args.dimension; ++i) {
        lower[i] = benchmark_functions_list[id].global_min[0];
        upper[i] = benchmark_functions_list[id].global_min[1];
        solution[i] = benchmark_functions_list[id].solution;
    }

    return {args.dimension, lower, upper, solution, benchmark_functions_list[id].solving_function,
            args.tolerance, args.population_size, args.max_generations};
}