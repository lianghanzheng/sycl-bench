#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Gramschmidt1;
class Gramschmidt2;
class Gramschmidt3;

void init_array(DATA_TYPE* A, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t i = 0; i < M; i++) {
		for(size_t j = 0; j < N; j++) {
			A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
		}
	}
}

void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t k = 0; k < N; k++) {
		DATA_TYPE nrm = 0;
		for(size_t i = 0; i < M; i++) {
			nrm += A[i * N + k] * A[i * N + k];
		}

		R[k * N + k] = sqrt(nrm);
		for(size_t i = 0; i < M; i++) {
			Q[i * N + k] = A[i * N + k] / R[k * N + k];
		}

		for(size_t j = k + 1; j < N; j++) {
			R[k * N + j] = 0;
			for(size_t i = 0; i < M; i++) {
				R[k * N + j] += Q[i * N + k] * A[i * N + j];
			}
			for(size_t i = 0; i < M; i++) {
				A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
			}
		}
	}
}

class Polybench_Gramschmidt {
  public:
	Polybench_Gramschmidt(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		R.resize(size * size);
		Q.resize(size * size);

		init_array(A.data(), size);

		//A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		//R_buffer.initialize(args.device_queue, R.data(), cl::sycl::range<2>(size, size));
		//Q_buffer.initialize(args.device_queue, Q.data(), cl::sycl::range<2>(size, size));
		A_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
		R_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
		Q_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
		for (size_t i = 0; i < size*size; ++i) {
		  A_buffer[i] = A[i];
		  R_buffer[i] = R[i];
		  Q_buffer[i] = Q[i];
		}
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		for(size_t k = 0; k < size; k++) {
			auto e1 = args.device_queue.submit([&](handler& cgh) {
				//auto A = A_buffer.get_access<access::mode::read>(cgh);
				//auto R = R_buffer.get_access<access::mode::write>(cgh);
				DATA_TYPE *A = A_buffer;
				DATA_TYPE *R = R_buffer;
				cgh.parallel_for<Gramschmidt1>(range<2>(1, 1), [=, M_ = size](item<2> item) {
					DATA_TYPE nrm = 0;
					for(size_t i = 0; i < M_; i++) {
						nrm += A[i*M_ + k] * A[i*M_ + k];
					}
					R[k*M_ + k] = cl::sycl::sqrt(nrm);
				});
			});
			events.push_back(e1);

			auto e2 = args.device_queue.submit([&](handler& cgh) {
				//auto A = A_buffer.get_access<access::mode::read>(cgh);
				//auto R = R_buffer.get_access<access::mode::read>(cgh);
				//auto Q = Q_buffer.get_access<access::mode::write>(cgh);
				DATA_TYPE *A = A_buffer;
				DATA_TYPE *R = R_buffer;
				DATA_TYPE *Q = Q_buffer;
				cgh.depends_on(e1);
				cgh.parallel_for<Gramschmidt2>(range<2>(size, 1), id<2>(0, k), [=, N=size](item<2> item) { 
					const auto i = item[0];
					const auto j = item[1];

					Q[i*N+j] = A[i*N+j] / R[k*N + k]; 
				});
			});
			events.push_back(e2);

			auto e3 = args.device_queue.submit([&](handler& cgh) {
				//auto A = A_buffer.get_access<access::mode::read_write>(cgh);
				//auto R = R_buffer.get_access<access::mode::write>(cgh);
				//auto Q = Q_buffer.get_access<access::mode::read>(cgh);
				DATA_TYPE *A = A_buffer;
				DATA_TYPE *R = R_buffer;
				DATA_TYPE *Q = Q_buffer;
				cgh.depends_on({e1, e2});
				cgh.parallel_for<Gramschmidt3>(range<2>(size, 1), [=, M_ = size, N_ = size](item<2> item) {
					const auto j = item[0];
					const auto jj = item[1];

					if(j <= k || j >= N_) return;

					R[j*M_ + jj] = 0;
					for(size_t i = 0; i < M_; i++) {
						R[j*M_ + jj] += Q[i*M_ + k] * A[i*M_ + j];
					}

					for(size_t i = 0; i < M_; i++) {
						A[i*M_ + j] -= Q[i*M_ + k] * R[j*M_ + jj];
					}
				});
			});
			e3.wait();
			events.push_back(e3);
		}
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> A_cpu(size * size);
		std::vector<DATA_TYPE> R_cpu(size * size);
		std::vector<DATA_TYPE> Q_cpu(size * size);

		// Trigger writeback
		//A_buffer.reset();

		init_array(A_cpu.data(), size);

		gramschmidt(A_cpu.data(), R_cpu.data(), Q_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(A_cpu[i * size + j], A_buffer[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Gramschmidt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> R;
	std::vector<DATA_TYPE> Q;

	//PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	//PrefetchedBuffer<DATA_TYPE, 2> R_buffer;
	//PrefetchedBuffer<DATA_TYPE, 2> Q_buffer;
	DATA_TYPE *A_buffer;
	DATA_TYPE *R_buffer;
	DATA_TYPE *Q_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Gramschmidt>();
	return 0;
}
