#include <iostream>
#include <ica/benchmarks/ICABenchmarkFunctions.h>
#include <ica/solver_serial.h>
#include <ica/solver_smp.h>
#include <ica/solver_opencl.h>

int main(int argc, char **args) {
    auto cfg = solver::load_config("../configs/ica.cfg");

    for (size_t i = 0; i < benchmark::BenchmarkFunc::TOTAL_FUNCTIONS; ++i) {
        auto setup = benchmark::BenchmarkFunc::create_solver(cfg, i);
        std::cout << "Benchmark function = " << benchmark::BenchmarkFunc::get_func_name(i) << "\n";
        if (cfg.methods[0]) {
            solve_serial(setup);
        }
        if (cfg.methods[1]) {
            solve_smp(setup);
        }
        if (cfg.methods[2]) {
            solve_opencl(setup);
        }
    }

    return 0;
}
