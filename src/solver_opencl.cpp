#include <ica/solver_opencl.h>
#include <ica/imperialist_alg.h>

RESULT_FLAGS solve_opencl(solver::ICASolver_t &setup) {
	if (setup.dimension == 0) return INVALID_ARGS;
	if (setup.population_size == 0) return FAIL;

	alg::ICAOpenCL ica(setup);
	bool res = ica.run();

	return res ? OK : FALSE;
}