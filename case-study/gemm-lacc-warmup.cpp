#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#define ALPHA 32412
#define BETA 2123

using DATA_TYPE = float;

class Gemm;

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NK; j++) {
			A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(size_t i = 0; i < NK; i++) {
		for(size_t j = 0; j < NJ; j++) {
			B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
		}
	}

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
		}
	}
}

void gemm(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			C[i * NJ + j] *= BETA;

			for(size_t k = 0; k < NK; ++k) {
				C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
			}
		}
	}
}

class Polybench_Gemm {
  public:
	Polybench_Gemm(const BenchmarkArgs& args) 
	    : args(args), size(args.problem_size), tilesize(args.local_size) {
	  A_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
	  B_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
	  C_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
	}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);

		init(A.data(), B.data(), C.data(), size);

		//A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		//B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
		//C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
		DATA_TYPE *A_h = A.data();
		DATA_TYPE *B_h = B.data();
		DATA_TYPE *C_h = C.data();
		DATA_TYPE *A_d = A_buffer;
		DATA_TYPE *B_d = B_buffer;
		DATA_TYPE *C_d = C_buffer;

		cl::sycl::range<2> G {size, size};
		cl::sycl::range<2> L {1, tilesize};
		args.device_queue.submit([&](cl::sycl::handler &cgh) {
		  cgh.parallel_for(
		      cl::sycl::nd_range {G, L},
		      [=, N=size, L=tilesize](cl::sycl::nd_item<2> item) {
		    const auto i = item.get_global_id(0);
		    const auto j = item.get_global_id(1);

		    A_d[i*N+j] = A_h[i*N+j];
		    B_d[i*N+j] = B_h[i*N+j];
		    C_d[i*N+j] = C_h[i*N+j];
		  });
		}).wait();
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		range<2> G {size, size};
		range<2> L {1, tilesize};
		events.push_back(args.device_queue.submit([&](handler& cgh) {
		        DATA_TYPE *A = A_buffer;
		        DATA_TYPE *B = B_buffer;
		        DATA_TYPE *C = C_buffer;
			local_accessor<DATA_TYPE, 1> tileA(tilesize, cgh);

			cgh.parallel_for<Gemm>(nd_range<2> {G, L}, [=, NK_=size, NL_=tilesize](nd_item<2> item) {
				const auto i = item.get_global_id(0);
				const auto j = item.get_global_id(1);
				const auto li = item.get_local_id(1);

				C[i*NK_ + j] *= BETA;

				for (size_t k = 0; k < NK_; k+=NL_) {
				  tileA[li] = A[i*NK_ + k+li];
				  group_barrier(item.get_group());

				  for (size_t kk = 0; kk < NL_; ++kk) {
				    C[i*NK_ + j] += ALPHA * tileA[kk] * B[(k+kk)*NK_ + j];
				  }
				  group_barrier(item.get_group());
				}
				//for(size_t k = 0; k < NK_; k++) {
				//	C[{i, j}] += ALPHA * A[{i, k}] * B[{k, j}];
				//}
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		// Trigger writeback
		//C_buffer.reset();

		std::vector<DATA_TYPE> C_cpu(size * size);

		init(A.data(), B.data(), C_cpu.data(), size);

		gemm(A.data(), B.data(), C_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(C_cpu[i * size + j], C_buffer[i*size + j]);
				if(diff > ERROR_THRESHOLD) {
				  printf("%f, %f\n", C_cpu[i * size + j], C_buffer[i*size + j]);
				  return false;
				}
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Gemm"; }

private:
	BenchmarkArgs args;

	const size_t size;
	const size_t tilesize;
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
	app.run<Polybench_Gemm>();
	return 0;
}
