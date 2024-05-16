#include "common.h"

namespace s = cl::sycl;

template <typename DataT, int Dims>
class MicroBenchDRAMKernel1;


template <typename DataT, int Dims>
class MicroBenchDRAMKernel2;


template <typename DataT, int Dims>
class CopyBufferDummyKernel;

template <typename DataT, int Dims>
s::range<Dims> getBufferSize(size_t problemSize) {
  if constexpr(Dims == 1) {
    return s::range<1>(problemSize * problemSize * problemSize / sizeof(DataT));
  }
  if constexpr(Dims == 2) {
    return s::range<2>(problemSize * problemSize / sizeof(DataT), problemSize);
  }
  if constexpr(Dims == 3) {
    return s::range<3>(problemSize / sizeof(DataT), problemSize, problemSize);
  }
}

/**
 * Microbenchmark measuring DRAM bandwidth.
 */
template <typename DataT, int Dims>
class MicroBenchDRAM {
protected:
  BenchmarkArgs args;
  std::vector<DataT> input;
  
  size_t USM_size;
  DataT *input_usm;
  DataT *output_usm;


public:
  MicroBenchDRAM(const BenchmarkArgs& args):args(args),USM_size(getBufferSize<DataT, Dims>(args.problem_size).size())
  {
        input.resize(USM_size,33.f);
  }

  void setup() {
    input_usm = s::malloc_device<DataT>(USM_size,args.device_queue);
    output_usm = s::malloc_shared<DataT>(USM_size,args.device_queue);
    // Bind data to their cores with parallel_for.
    args.device_queue.submit([&](cl::sycl::handler& h){
        //h.memcpy(input_usm, &input[0], USM_size*sizeof(DataT));
        auto in_d = input_usm;
        auto in_h = &input[0];
        auto out_s = output_usm;
        h.parallel_for(USM_size, [=, N=USM_size](s::id<1> i) {
          in_d[i] = in_h[i];
	        out_s[i] = 0.0f;
        });
    }).wait();
  }

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    const double copiedGiB =
        getBufferSize<DataT, Dims>(args.problem_size).size() * sizeof(DataT) / 1024.0 / 1024.0 / 1024.0;
    // Multiply by two as we are both reading and writing one element in each thread.
    return {copiedGiB * 2.0, "GiB"};
  }

  void run(std::vector<s::event>& events) {    
    auto in = input_usm;
    auto out = output_usm;

    events.push_back(args.device_queue.parallel_for(
        USM_size, [=, N = USM_size](s::id<1> i) {
      out[i] = in[i];
    }));
  }


  bool verify(VerificationSetting& ver) {
    bool flag = true;
    for(size_t i = 0; i < USM_size; ++i) {
      if(output_usm[i]!=33.f)
      {
        flag = false;
        break;
      }
    }

    free(input_usm,args.device_queue);
    free(output_usm,args.device_queue);

    return flag;
  }
  
  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "MicroBench_DRAM_CPU_";
    name << ReadableTypename<DataT>::name;
    name << "_" << Dims;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchDRAM<float, 1>>();
  if(app.deviceSupportsFP64()) {
    app.run<MicroBenchDRAM<double, 1>>();
  }

  return 0;
}

