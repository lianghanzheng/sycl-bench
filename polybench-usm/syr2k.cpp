#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Syr2k1;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void init_arrays(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
		}

		for(size_t j = 0; j < M; j++) {
			A[i * N + j] = ((DATA_TYPE)i * j) / N;
			B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
		}
	}
}

void syr2k(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			C[i * N + j] *= BETA;
		}
	}

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			for(size_t k = 0; k < M; k++) {
				C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
				C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
			}
		}
	}
}

class Polybench_Syr2k {
  public:
	Polybench_Syr2k(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);

		init_arrays(A.data(), B.data(), C.data(), size);

		//A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		//B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
		//C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
		A_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
		B_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
		C_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
		for (size_t i = 0; i < size*size; ++i) {
		  A_buffer[i] = A[i];
		  B_buffer[i] = B[i];
		  C_buffer[i] = C[i];
		}
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			//auto A = A_buffer.get_access<access::mode::read>(cgh);
			//auto B = B_buffer.get_access<access::mode::read>(cgh);
			//auto C = C_buffer.get_access<access::mode::read_write>(cgh);
			DATA_TYPE *A = A_buffer;
			DATA_TYPE *B = B_buffer;
			DATA_TYPE *C = C_buffer;
			cgh.parallel_for<Syr2k1>(range<2> {size, size}, [=, M_ = size](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				C[i*M_ + j] *= BETA;

				for(size_t k = 0; k < M_; k++) {
					C[i*M_+j] += ALPHA * A[i*M_ + k] * B[j*M_ + k] + ALPHA * B[i*M_ + k] * A[j*M_ + k];
				}
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> C_cpu(size * size);

		init_arrays(A.data(), B.data(), C_cpu.data(), size);

		// Trigger writeback
		//C_buffer.reset();

		syr2k(A.data(), B.data(), C_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(C_cpu[i * size + j], C_buffer[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Syr2k"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> C;

	//PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	//PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	//PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
	DATA_TYPE *A_buffer;
	DATA_TYPE *B_buffer;
	DATA_TYPE *C_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Syr2k>();
	return 0;
}
