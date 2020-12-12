//
// Created by Patrik Patera.
//

#include <ica/iface/ICASolver.h>

solver::ICAArgs solver::load_config(const std::string &config_path) {
    libconfig::Config config;
    try {
        config.readFile(config_path.c_str());
    } catch (const libconfig::FileIOException &err) {
        std::cout << "Error while reading config file!\n";
        return default_args;
    } catch (const libconfig::ParseException &err) {
        std::cout << "Error while parsing config file!\n";
        return default_args;
    }

    double tolerance = DEFAULT_TOLERANCE;
    uint dim = DEFAULT_DIMENSION, pop = DEFAULT_POPULATION_SIZE, gen = static_cast<uint>(DEFAULT_MAX_GENERATIONS);
    bool methods[3] = {true, true, false};

    config.lookupValue("dimension", dim);
    config.lookupValue("population_size", pop);
    config.lookupValue("max_generations", gen);
    config.lookupValue("tolerance", tolerance);
    config.lookupValue("methods.serial", methods[0]);
    config.lookupValue("methods.smp", methods[1]);
    config.lookupValue("methods.opencl", methods[2]);



    return {dim, pop, gen, tolerance, {methods[0], methods[1], methods[2]}};
}



