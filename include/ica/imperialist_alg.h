#pragma once

#include <algorithm>
#include <numeric>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <tbb/tbb.h>
#include <CL/cl2.hpp>

#include <ica/iface/ICASolver.h>


namespace alg {

	/* Forward imperialist structure declaration. */
	struct imperialist_t;

	/*
	  Structure represents colony.
	*/
	struct  colony_t {
		double cost;	// Actual cost of colony.

		struct imperialist_t* imp;	// Pointer to appropriate imperialist.

		std::vector<double> values;		// Values in n-dimension space (problem size).

		/*
		  Creates colony with input dimension, boundaries and PRNG (mt19937_64).
		  Generating also random values in a space.
		*/
		explicit colony_t(size_t dim, const double* lower, const double* upper, std::mt19937_64& mt) : cost(DBL_MAX), imp(nullptr), values(dim)
		{
			generate_rand_values(lower, upper, mt);
		}

		/*
		  Creates colony based on input cost, imperialist (pointer) and values.
		*/
		colony_t(double _cost, imperialist_t* _imp, std::vector<double>& _values) : cost(_cost), imp(_imp), values(_values) {}

		/*
		  Creates colony with appropriate imperialist pointer and generated values.
		*/
		colony_t(imperialist_t* _imp, size_t dim, const double* lower, const double* upper, std::mt19937_64& mt) : cost(DBL_MAX), imp(_imp), values(dim)
		{
			generate_rand_values(lower, upper, mt);
		}

		/*
		  Generating random numbers in input boundaries and specific PRNG (mt19937_64).
		*/
		inline void generate_rand_values(const double *l, const double *u, std::mt19937_64& mt) {
			for (size_t i = 0; i < values.size(); ++i) {
				std::uniform_real_distribution<double> distr(l[i], u[i]);
				values[i] = distr(mt);
			}
		}
	};

	/*
	  Structure represents imperialist.
	*/
	struct imperialist_t {
		double cost, power, prob_possesion;	// Fitness value; total power; colonies probability of possesion
		size_t N_c;	// Number of colonies while initialization.

		std::vector<size_t> colonies;	// Indicies of colonies (in colonies array) belong to that imperialist.
		std::vector<double> values;		// Values in n-dimension space (problem size).

		/*
		  Creates imperialist from a colony.
		*/
		imperialist_t(const colony_t& col) : cost(col.cost), power(0.), prob_possesion(0.), N_c(0), colonies(0), values(col.values) {}
	};

	/* Comparing colonies according to their fitness value - ascending */
	inline bool operator<(const colony_t& c1, const colony_t& c2) { return c1.cost < c2.cost; }

	/* Comparing colonies according to their fitness value  - descending */
	inline bool operator>(const colony_t& c1, const colony_t& c2) { return c1.cost > c2.cost; }

	/*
	  Abstract class with pure virtual method - run().
	  There is full implementation of serial version, because of using these methods in other versions.
	  Only move_colonies and compute_colonies_mean are overridable (different implementation for parallel versions).
	*/
	class IICA {
	protected:
		const solver::ICASolver_t& m_solver;

		static thread_local std::uniform_real_distribution<double> uni;	// Uniform distribution in [0, 1) interval.
		static thread_local std::mt19937_64 mt;	// PRNG 
		static uint64_t cl_seed[4];	// Seed for OpenCL PRNG (Xoroshiro256+).

		double best_cost, revolution_rate;	// Actual the best fitness value; actual revolution rate in revolution process.

		size_t N_pop, N_imp;	// Number of total population; number of total imperialist.

		std::vector<colony_t> colonies_vec;		// Array with all colonies.
		std::vector<std::unique_ptr<imperialist_t>> imperialist_vec;	// Array with all imperialists.

		static const double beta, dump_revolution_rate, xi, init_revolution_rate;		// Hyper-parameters using in ICA.
		static const size_t max_repetition, mt_seed;	// Max. repetition of same fitness value for stop condition; seed for PRNG - mt - in this class;


		/*
		  Compute mean value (average) of all colonies according to their fitness value.
		*/
		virtual double compute_colonies_mean(std::vector<size_t>& col_indicies) const;

		/*
		  Move colonies close to its imperialist.
		*/
		virtual void move_colonies();

		/*
		  Stop condition based on repetition of same best fitness value or max. generation value.
		*/
		bool stop_condition(double A, double B, double max_diff, size_t& rep, size_t it);

		/*
		  Swap values and fitness value between colony and imperialist. Colony will become imperialist.
		*/
		void swap_colony_imperialist(colony_t& col, imperialist_t* imp);

		/*
		  Compute total power of all empires for imperialist competition.
		*/
		size_t compute_total_empire_power();

		/*
		  Change position colony with imperialist.
		*/
		void change_position();

		/*
		  Imperialist competition - winner will get colony.
		  Weakest parameter is the colony for competition (index to colonies array).
		*/
		void imperialist_competition(size_t weakest);

		/*
		  Eliminate all empires which does not have any colony.
		*/
		void eliminate_empire();

		/*
		  Revolution of all colonies. Each empire has revolution probability.
		  Some colonies will get new random values in space.
		*/
		void do_revolution();
	public:
		explicit IICA(const solver::ICASolver_t& setup);

		/*
		  Run the algorithm.
		*/
		virtual bool run() = 0;
	};

	/*
	  ICA Serial version clas.
	*/
	class ICASerial : public IICA {
	public:
		explicit ICASerial(const solver::ICASolver_t& setup);

		bool run() override;
	};

	/*
	  ICA SMP version clas.
	*/
	class ICASMP : public IICA {
	protected:
		double compute_colonies_mean(std::vector<size_t>& col_indicies) const override;

		void move_colonies() override;

	public:
		explicit ICASMP(const solver::ICASolver_t& setup);

		bool run() override;
	};

	/*
	  ICA OpenCL version clas.
	*/
	class ICAOpenCL : public IICA {
	private:
		cl::Program pr_moving_colonies;	    // OpenCL Program with moving colonies.
		cl::Platform default_platform;	    // OpenCL Platform.
		cl::Device default_device;		    // OpenCL Device.
		cl::Context context;			    // OpenCL Context.

		bool is_prepared;				// Flag if everything is prepared for running.s

		/* Source code of moving colonies for OpenCL */
		static constexpr auto move_colonies_src =
			"double rand(__global ulong* s) {"
			" const ulong result = s[0] + s[3];"
			" const ulong t = s[1] << 17;"
			" s[2] ^= s[0];"
			" s[3] ^= s[1];"
			" s[1] ^= s[2];"
			" s[0] ^= s[3];"
			" s[2] ^= t;"
			" s[3] = (s[3] << 35) | (s[3] >> (64 - 45));"
			" return ((double)result / ULONG_MAX);"
			"}"
			" "
			"__kernel void move_colonies(__global ulong* col_idx, __global double* col, __global double* imp, __global ulong* seed, __global double* upper, __global double* lower, const double beta, const ulong col_s, const ulong dim) { "
			"const int idx = get_global_id(0); "
			" if (idx < col_s) {"
			"   const ulong i = idx * dim;"
			"   for (ulong j = 0; j < dim; ++j) { "
			"		col[i + j] += (beta * rand(seed) - 1) * (imp[col_idx[idx] * dim + j] - col[i + j]);"
			" "
			"       if (col[i + j] > upper[j]) {"
			"         double e = upper[j] - lower[j];"
			"         col[i + j] = 2.0 * upper[j] - (col[i + j] - fabs((col[i + j] - upper[j]) / e) * e);"
			"       } else if (col[i + j] <= lower[j]) {"
			"         double e = upper[j] - lower[j];"
			"         col[i + j] = 2.0 * lower[j] - (col[i + j] + fabs((lower[j] - col[i + j]) / e) * e);"
			"       }"
			"    }"
			" }"
			"}";

		/*
		  Prepare OpenCL Program, context, platform and device for running a kernel function.
		*/
		bool opencl_prepare();

		/*
		  Compute fitness function of all colonies.
		*/
		void compute_fitness();

	protected:
		/*
		  Copy args and run kernel function on GPU using OpenCL.
		*/
		void move_colonies() override;

	public:
		explicit ICAOpenCL(const solver::ICASolver_t& setup);

		bool run() override;
	};
}
