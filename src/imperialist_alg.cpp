#include <ica/imperialist_alg.h>

const double alg::IICA::beta = 3.39;

const double alg::IICA::dump_revolution_rate = .99;

const double alg::IICA::xi = 0.1;

const double alg::IICA::init_revolution_rate = 0.33;

const size_t alg::IICA::max_repetition = 1000;

const size_t alg::IICA::mt_seed = size_t(78e14);

thread_local std::mt19937_64 alg::IICA::mt = std::mt19937_64(size_t(alg::IICA::mt_seed));

thread_local std::uniform_real_distribution<double> alg::IICA::uni = std::uniform_real_distribution<double>(0., 1.);

uint64_t alg::IICA::cl_seed[4] = {345, 6436346, 456545, 6456436};

double alg::IICA::compute_colonies_mean(std::vector<size_t> &col_indicies) const {
    if (col_indicies.size() == 0) return 0.;

    double sum = 0.;
    for (const auto i : col_indicies) sum += colonies_vec[i].cost;
    return (sum / col_indicies.size());
}

void alg::IICA::swap_colony_imperialist(colony_t &col, imperialist_t *imp) {
    std::swap<double>(col.cost, imp->cost);
    col.values.swap(imp->values);
}

alg::IICA::IICA(const solver::ICASolver_t &setup) : N_pop(setup.population_size), m_solver(setup),
                                                    revolution_rate(init_revolution_rate), best_cost(DBL_MAX) {
    mt = std::mt19937_64(mt_seed);
    uni = std::uniform_real_distribution<double>(0., 1.);

    N_imp = setup.population_size >= size_t(50) ? size_t(5) : size_t(3);

    colonies_vec.reserve(N_pop);
    imperialist_vec.reserve(N_imp);

    // Generate initial population (countries).
    for (size_t i = 0; i < N_pop; ++i) {
        colonies_vec.emplace_back(setup.dimension, setup.lower_bound, setup.upper_bound, mt);
        colonies_vec[i].cost = setup.solving_function(colonies_vec[i].values.data(), setup.dimension);
    }

    std::partial_sort(colonies_vec.rbegin(), colonies_vec.rbegin() + N_imp, colonies_vec.rend(),
                      [](const auto &c1, const auto &c2) { return c1 < c2; });

    for (size_t i = 0; i < N_imp; ++i) {
        auto imp = colonies_vec.back();
        colonies_vec.pop_back();
        imperialist_vec.emplace_back(std::make_unique<imperialist_t>(imp));
    }

    imperialist_vec.shrink_to_fit();

    // Normalize power of each imperialist by the most expensive cost value.
    const double max_cost_imp = imperialist_vec[N_imp - 1]->cost;
    double accumulated_power = 0.0;

    for (size_t i = 0; i < N_imp; ++i) {
        imperialist_vec[i]->power = max_cost_imp - imperialist_vec[i]->cost;
        accumulated_power += imperialist_vec[i]->power;
    }

    // Compute number of colonies and the final power of each imperialist (heigher cost = lower power).
    size_t cntr = 0, N_col = N_pop - N_imp;
    for (size_t i = 0; i < N_imp; ++i) {
        imperialist_vec[i]->power = imperialist_vec[i]->power / accumulated_power;
        imperialist_vec[i]->N_c = static_cast<size_t>(imperialist_vec[i]->power * N_col);
        cntr += imperialist_vec[i]->N_c;
    }
    imperialist_vec[N_imp - 1]->N_c += (N_col - cntr);

    size_t init_col = 0;
    for (size_t i = 0; i < N_imp; ++i) {
        for (size_t j = 0; j < imperialist_vec[i]->N_c; ++j) {
            imperialist_vec[i]->colonies.push_back(init_col);
            colonies_vec[init_col].imp = imperialist_vec[i].get();
            ++init_col;
        }
    }
}

size_t alg::IICA::compute_total_empire_power() {
    // Compute the empire power and find maximal one.
    double max_e_p = 0.;
    for (size_t i = 0; i < imperialist_vec.size(); ++i) {
        double col_mean = compute_colonies_mean(imperialist_vec[i]->colonies);
        imperialist_vec[i]->power = imperialist_vec[i]->cost + xi * col_mean;
        if (max_e_p < imperialist_vec[i]->power) max_e_p = imperialist_vec[i]->power;
    }

    // Normalize the empire power by max value.
    double acc_e_p = 0.;
    for (size_t i = 0; i < imperialist_vec.size(); ++i) {
        imperialist_vec[i]->power = max_e_p - imperialist_vec[i]->power;
        acc_e_p += imperialist_vec[i]->power;
    }

    // Compute probability of possession of colonies.
    double weakest_imp_val = 0.;
    size_t weakest_imp_idx = 0;
    for (size_t i = 0; i < imperialist_vec.size(); ++i) {
        imperialist_vec[i]->prob_possesion = imperialist_vec[i]->power / acc_e_p;
        if (imperialist_vec[i]->power > weakest_imp_val) {
            weakest_imp_val = imperialist_vec[i]->power;
            weakest_imp_idx = i;
        }
    }

    if (imperialist_vec[weakest_imp_idx]->colonies.empty()) return colonies_vec.size();

    // Find the weakest colony in the weakest empire and return appropriate index of colony.
    double tmp_weak = colonies_vec[imperialist_vec[weakest_imp_idx]->colonies[0]].cost;
    size_t weakest_col_idx = 0, pos = 0;
    for (size_t i = 0; i < imperialist_vec[weakest_imp_idx]->colonies.size(); ++i) {
        if (colonies_vec[imperialist_vec[weakest_imp_idx]->colonies[i]].cost >= tmp_weak) {
            pos = i;
            weakest_col_idx = imperialist_vec[weakest_imp_idx]->colonies[i];
        }
    }

    // Delete the weakest colony from the weakest empire.
    imperialist_vec[weakest_imp_idx]->colonies.erase(imperialist_vec[weakest_imp_idx]->colonies.begin() + pos);

    return weakest_col_idx;
}

void alg::IICA::move_colonies() {
    for (size_t i = 0; i < colonies_vec.size(); ++i) {
        auto &col = colonies_vec[i];

        for (size_t j = 0; j < col.values.size(); ++j) {
            col.values[j] += (beta * uni(mt) - 1) * (col.imp->values[j] - col.values[j]);

            if (col.values[j] > m_solver.upper_bound[j]) {
                col.values[j] = 2.0 * m_solver.upper_bound[j] -
                                (col.values[j] - std::abs((col.values[j] - m_solver.upper_bound[j])
                                                          / (m_solver.upper_bound[j] - m_solver.lower_bound[j])) *
                                                 (m_solver.upper_bound[j] - m_solver.lower_bound[j]));
            } else if (col.values[j] < m_solver.lower_bound[j]) {
                col.values[j] = 2.0 * m_solver.lower_bound[j] -
                                (col.values[j] + std::abs((m_solver.lower_bound[j] - col.values[j])
                                                          / (m_solver.upper_bound[j] - m_solver.lower_bound[j])) *
                                                 (m_solver.upper_bound[j] - m_solver.lower_bound[j]));
            }
        }

        col.cost = m_solver.solving_function(col.values.data(), m_solver.dimension);
    }
}

void alg::IICA::change_position() {
    for (size_t i = 0; i < colonies_vec.size(); ++i) {
        if (colonies_vec[i].cost < best_cost) best_cost = colonies_vec[i].cost;
        if (colonies_vec[i].cost < colonies_vec[i].imp->cost) {
            swap_colony_imperialist(colonies_vec[i], colonies_vec[i].imp);
        }
    }
}

void alg::IICA::imperialist_competition(size_t weakest) {
    if (weakest == colonies_vec.size()) return;

    double max = -2;
    size_t pos = 0;

    for (size_t i = 0; i < imperialist_vec.size(); ++i) {
        if (colonies_vec[weakest].imp == imperialist_vec[i].get()) continue;

        double prob_pos = imperialist_vec[i]->prob_possesion - uni(mt);
        if (prob_pos > max) {
            max = prob_pos;
            pos = i;
        }
    }

    colonies_vec[weakest].imp = imperialist_vec[pos].get();
    imperialist_vec[pos]->colonies.push_back(weakest);
}

void alg::IICA::eliminate_empire() {
    for (auto it = imperialist_vec.begin(); it != imperialist_vec.end();) {
        if ((*it)->colonies.size() == 0) {
            //colonies_vec.emplace_back((*it)->cost, (*it).get(), (*it)->values);
            // Generate new values in space.
            colonies_vec.emplace_back((*it).get(), m_solver.dimension, m_solver.lower_bound, m_solver.upper_bound, mt);
            imperialist_competition(colonies_vec.size() - 1);
            it = imperialist_vec.erase(it);
        } else {
            ++it;
        }
    }
}

void alg::IICA::do_revolution() {
    for (size_t i = 0; i < imperialist_vec.size(); ++i) {
        size_t n = static_cast<size_t>(std::floor(revolution_rate * imperialist_vec[i]->colonies.size()));
        std::uniform_int_distribution<size_t> rnd(1, imperialist_vec[i]->colonies.size());

        for (size_t j = 0; j < n; ++j) {
            colonies_vec[imperialist_vec[i]->colonies[rnd(mt) - 1]].generate_rand_values(m_solver.lower_bound,
                                                                                         m_solver.upper_bound, mt);
        }
    }
}

bool alg::IICA::stop_condition(double A, double B, double max_diff, size_t &rep, size_t it) {
    if (std::fabs(A - B) <= max_diff) ++rep;
    else rep = 0;

    return (rep < max_repetition) && (it < m_solver.max_generations);
}

/*-------ICA SERIAL VERSION--------*/

alg::ICASerial::ICASerial(const solver::ICASolver_t &setup) : IICA(setup) {}

bool alg::ICASerial::run() {
    size_t iter = 0, rep = 0;
    double actual_cost = DBL_MAX;

    std::cout << "Computing - Serial version\n";
    while (stop_condition(actual_cost, best_cost, m_solver.tolerance, rep, iter)) {
        actual_cost = best_cost;

        do_revolution();
        move_colonies();
        change_position();

        if (imperialist_vec.size() > 1) {
            size_t weakest_col_idx = compute_total_empire_power();
            imperialist_competition(weakest_col_idx);
            eliminate_empire();
        }

        revolution_rate *= dump_revolution_rate;
        ++iter;
    }

    if (imperialist_vec.size() > 1)
        std::partial_sort(imperialist_vec.begin(), imperialist_vec.begin() + 1, imperialist_vec.end(),
                          [](const auto &c1, const auto &c2) { return c1->cost < c2->cost; });

    std::copy(imperialist_vec[0]->values.begin(), imperialist_vec[0]->values.end(), m_solver.solution);
    std::cout << "\t Cost = " << imperialist_vec[0]->cost << "\n";
    std::cout << "-------------------------------------\n";

    return true;
}

/*-------ICA SMP VERSION--------*/

double alg::ICASMP::compute_colonies_mean(std::vector<size_t> &col_indicies) const {
    if (col_indicies.size() == 0) return 0.;

    double sum = tbb::parallel_reduce(
            tbb::blocked_range<std::vector<size_t>::iterator>(col_indicies.begin(), col_indicies.end()), double(0),
            [&](const tbb::blocked_range<std::vector<size_t>::iterator> &it, double init) {
                return std::accumulate(it.begin(), it.end(), init,
                                       [&](double val, size_t j) { return val + colonies_vec[j].cost; });
            }, [](double x, double y) { return x + y; }, tbb::simple_partitioner());

    return (sum / col_indicies.size());
}

void alg::ICASMP::move_colonies() {
    tbb::parallel_for(size_t(0), colonies_vec.size(), [&](size_t i) {
        auto &col = colonies_vec[i];

        for (size_t j = 0; j < col.values.size(); ++j) {
            col.values[j] += (beta * uni(mt) - 1) * (col.imp->values[j] - col.values[j]);

            if (col.values[j] > m_solver.upper_bound[j]) {
                col.values[j] = 2.0 * m_solver.upper_bound[j] -
                                (col.values[j] - std::abs((col.values[j] - m_solver.upper_bound[j])
                                                          / (m_solver.upper_bound[j] - m_solver.lower_bound[j])) *
                                                 (m_solver.upper_bound[j] - m_solver.lower_bound[j]));
            } else if (col.values[j] < m_solver.lower_bound[j]) {
                col.values[j] = 2.0 * m_solver.lower_bound[j] -
                                (col.values[j] + std::abs((m_solver.lower_bound[j] - col.values[j])
                                                          / (m_solver.upper_bound[j] - m_solver.lower_bound[j])) *
                                                 (m_solver.upper_bound[j] - m_solver.lower_bound[j]));
            }
        }

        col.cost = m_solver.solving_function(col.values.data(), m_solver.dimension);
    }, tbb::simple_partitioner());
}

alg::ICASMP::ICASMP(const solver::ICASolver_t &setup) : IICA(setup) {}

bool alg::ICASMP::run() {
    size_t iter = 0, rep = 0;
    double actual_cost = DBL_MAX;

    std::cout << "Computing - SMP version\n";
    while (stop_condition(actual_cost, best_cost, m_solver.tolerance, rep, iter)) {
        actual_cost = best_cost;

        do_revolution();
        move_colonies();
        change_position();

        if (imperialist_vec.size() > 1) {
            size_t weakest_col = compute_total_empire_power();
            imperialist_competition(weakest_col);
            eliminate_empire();
        }

        revolution_rate *= dump_revolution_rate;
        ++iter;
    }

    if (imperialist_vec.size() > 1)
        std::partial_sort(imperialist_vec.begin(), imperialist_vec.begin() + 1, imperialist_vec.end(),
                          [](const auto &c1, const auto &c2) { return c1->cost < c2->cost; });

    std::copy(imperialist_vec[0]->values.begin(), imperialist_vec[0]->values.end(), m_solver.solution);
    std::cout << "\t Cost = " << imperialist_vec[0]->cost << "\n";
    std::cout << "-------------------------------------\n";

    return true;
}

/*-------ICA OPENCL VERSION--------*/

bool alg::ICAOpenCL::opencl_prepare() {
    cl::Program::Sources sources(1, std::string(move_colonies_src));
    cl::Platform::get(&default_platform);

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    if (all_devices.size() == 0) {
        return false;
    }

    default_device = all_devices[0];
    context = cl::Context({default_device});

    pr_moving_colonies = cl::Program(context, sources);

    return pr_moving_colonies.build({default_device}) == CL_SUCCESS;
}

void alg::ICAOpenCL::compute_fitness() {
    for (size_t i = 0; i < colonies_vec.size(); ++i) {
        colonies_vec[i].cost = m_solver.solving_function(colonies_vec[i].values.data(), m_solver.dimension);
    }
}

void alg::ICAOpenCL::move_colonies() {
    cl::CommandQueue que(context, default_device);

    cl::Buffer buff_col(context, CL_MEM_READ_WRITE, sizeof(double) * m_solver.dimension * colonies_vec.size());
    cl::Buffer buff_imp(context, CL_MEM_WRITE_ONLY, sizeof(double) * m_solver.dimension * imperialist_vec.size());

    /* Copy imperialist values into a continues buffer */
    for (size_t i = 0; i < imperialist_vec.size(); ++i) {
        que.enqueueWriteBuffer(buff_imp, CL_TRUE, i * sizeof(double) * m_solver.dimension,
                               sizeof(double) * m_solver.dimension,
                               imperialist_vec[i]->values.data());
    }

    /* Find appropriate indicies in imperialist vector */
    std::vector<size_t> tmp_col_idx;
    for (size_t i = 0; i < colonies_vec.size(); ++i) {
        for (size_t j = 0; j < imperialist_vec.size(); ++j) {
            if (imperialist_vec[j].get() == colonies_vec[i].imp) {
                tmp_col_idx.push_back(j);
                break;
            }
        }

        /* Copy colonies values into a continues buffer */
        que.enqueueWriteBuffer(buff_col, CL_TRUE, i * sizeof(double) * m_solver.dimension,
                               sizeof(double) * m_solver.dimension,
                               colonies_vec[i].values.data());
    }
    tmp_col_idx.shrink_to_fit();

    cl::Buffer buff_col_idx(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(size_t) * tmp_col_idx.size(),
                            tmp_col_idx.data());
    cl::Buffer buff_seed(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t) * 4, cl_seed);

    /* Copy lower and upper bounderies */
    cl::Buffer buff_upper(context, CL_MEM_WRITE_ONLY, sizeof(double) * m_solver.dimension);
    cl::Buffer buff_lower(context, CL_MEM_WRITE_ONLY, sizeof(double) * m_solver.dimension);
    que.enqueueWriteBuffer(buff_upper, CL_TRUE, 0, sizeof(double) * m_solver.dimension, m_solver.upper_bound);
    que.enqueueWriteBuffer(buff_lower, CL_TRUE, 0, sizeof(double) * m_solver.dimension, m_solver.lower_bound);

    /* Set kernel's arguments */
    cl::Kernel kernel(pr_moving_colonies, "move_colonies");
    kernel.setArg(0, buff_col_idx);
    kernel.setArg(1, buff_col);
    kernel.setArg(2, buff_imp);
    kernel.setArg(3, buff_seed);
    kernel.setArg(4, buff_upper);
    kernel.setArg(5, buff_lower);
    kernel.setArg(6, beta);
    kernel.setArg(7, colonies_vec.size());
    kernel.setArg(8, m_solver.dimension);

    que.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(colonies_vec.size()), cl::NullRange);
    que.finish();

    /* Copy back new seeds - only if seed param is __global ! */
    que.enqueueReadBuffer(buff_seed, CL_TRUE, 0, sizeof(uint64_t) * 4, cl_seed);

    /* Copy back new values to colonies */
    for (size_t i = 0; i < colonies_vec.size(); ++i) {
        que.enqueueReadBuffer(buff_col, CL_TRUE, i * sizeof(double) * m_solver.dimension,
                              sizeof(double) * m_solver.dimension,
                              colonies_vec[i].values.data());
    }
}

alg::ICAOpenCL::ICAOpenCL(const solver::ICASolver_t &setup) : IICA(setup) {
    is_prepared = opencl_prepare();
}

bool alg::ICAOpenCL::run() {
    if (!is_prepared) return false;

    size_t iter = 0, rep = 0;
    double actual_cost = DBL_MAX;

    std::cout << "Computing - OpenCL version\n";
    while (stop_condition(actual_cost, best_cost, m_solver.tolerance, rep, iter)) {
        actual_cost = best_cost;

        do_revolution();
        move_colonies();
        compute_fitness();
        change_position();

        if (imperialist_vec.size() > 1) {
            size_t weakest_col_idx = compute_total_empire_power();
            imperialist_competition(weakest_col_idx);
            eliminate_empire();
        }

        revolution_rate *= dump_revolution_rate;
        ++iter;
    }

    if (imperialist_vec.size() > 1)
        std::partial_sort(imperialist_vec.begin(), imperialist_vec.begin() + 1, imperialist_vec.end(),
                          [](const auto &c1, const auto &c2) { return c1->cost < c2->cost; });

    std::copy(imperialist_vec[0]->values.begin(), imperialist_vec[0]->values.end(), m_solver.solution);
    std::cout << "\t Cost = " << imperialist_vec[0]->cost << "\n";
    std::cout << "-------------------------------------\n";

    return true;
}
