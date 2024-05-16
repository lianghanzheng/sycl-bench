#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

class Atax1;
class Atax2;

void init_array(DATA_TYPE* x, DATA_TYPE* A, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t i = 0; i < NX; i++) {
		x[i] = i * M_PI;
		for(size_t j = 0; j < NY; j++) {
			A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
		}
	}
}

void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t i = 0; i < NX; i++) {
		for(size_t j = 0; j < NY; j++) {
			tmp[i] += A[i * NY + j] * x[j];
		}

		for(size_t j = 0; j < NY; j++) {
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}

class Polybench_Atax {
  public:
	Polybench_Atax(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		x.resize(size);
		y.resize(size);
		tmp.resize(size);

		init_array(x.data(), A.data(), size);

		//A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>{size, size});
		//x_buffer.initialize(args.device_queue, x.data(), cl::sycl::range<1>{size});
		//y_buffer.initialize(args.device_queue, y.data(), cl::sycl::range<1>{size});
		//tmp_buffer.initialize(args.device_queue, tmp.data(), cl::sycl::range<1>{size});
		A_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size*size, args.device_queue);
		x_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size, args.device_queue);
		y_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size, args.device_queue);
		tmp_buffer = cl::sycl::malloc_shared<DATA_TYPE>(size, args.device_queue);
		for (size_t i = 0; i < size*size; ++i) {
		  A_buffer[i] = A[i];
		}
		for (size_t i = 0; i < size; ++i) {
		  x_buffer[i] = x[i];
		  y_buffer[i] = y[i];
		  tmp_buffer[i] = tmp[i];
		}
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		auto e1 = args.device_queue.submit([&](handler& cgh) {
			//auto A = A_buffer.get_access<access::mode::read>(cgh);
			//auto x = x_buffer.get_access<access::mode::read>(cgh);
			//auto tmp = tmp_buffer.get_access<access::mode::read_write>(cgh);
			DATA_TYPE *A = A_buffer;
			DATA_TYPE *x = x_buffer;
			DATA_TYPE *tmp = tmp_buffer;
			cgh.parallel_for<Atax1>(range<1> {size}, [=, size_ = size](item<1> item) {
				const auto i = item[0];

				for(size_t j = 0; j < size_; j++) {
					tmp[i] += A[i*size_ + j] * x[j];
				}
			});
		});
		events.push_back(e1);

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			//auto A = A_buffer.get_access<access::mode::read>(cgh);
			//auto y = y_buffer.get_access<access::mode::read_write>(cgh);
			//auto tmp = tmp_buffer.get_access<access::mode::read>(cgh);
			DATA_TYPE *A = A_buffer;
			DATA_TYPE *y = y_buffer;
			DATA_TYPE *tmp = tmp_buffer;
			cgh.depends_on(e1);
			cgh.parallel_for<Atax2>(range<1> {size}, [=, size_ = size](item<1> item) {
				const auto j = item[0];

				for(size_t i = 0; i < size_; i++) {
					y[j] += A[i*size_ + j] * tmp[i];
				}
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		init_array(x.data(), A.data(), size);

		std::vector<DATA_TYPE> y_cpu(size);
		std::vector<DATA_TYPE> tmp_cpu(size);

		atax_cpu(A.data(), x.data(), y_cpu.data(), tmp_cpu.data(), size);

		//auto y_acc = y_buffer.get_access<cl::sycl::access::mode::read>();

		for(size_t i = 0; i < size; i++) {
			const auto diff = percentDiff(y_cpu[i], y_buffer[i]);
			if(diff > ERROR_THRESHOLD) return false;
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Atax"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> x;
	std::vector<DATA_TYPE> y;
	std::vector<DATA_TYPE> tmp;

	//PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	//PrefetchedBuffer<DATA_TYPE, 1> x_buffer;
	//PrefetchedBuffer<DATA_TYPE, 1> y_buffer;
	//PrefetchedBuffer<DATA_TYPE, 1> tmp_buffer;
	DATA_TYPE *A_buffer;
	DATA_TYPE *x_buffer;
	DATA_TYPE *y_buffer;
	DATA_TYPE *tmp_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Atax>();
	return 0;
}
