#ifndef DEEPVAN_BACKEND_OPENCL_HELPER_H_
#define DEEPVAN_BACKEND_OPENCL_HELPER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepvan/core/future.h"
#include "deepvan/core/runtime/opencl/cl2_header.h"
#include "deepvan/core/runtime/opencl/opencl_runtime.h"
#include "deepvan/core/runtime/opencl/opencl_shape_util.h"
#include "deepvan/core/types.h"
#include "deepvan/utils/macros.h"
#include "deepvan/utils/math.h"
#include "deepvan/utils/memory.h"

namespace deepvan {
// oorc for 'Out Of Range Check'
#define OUT_OF_RANGE_DEFINITION std::shared_ptr<BufferBase> oorc_flag;

#define OUT_OF_RANGE_CONFIG                                                    \
  if (runtime->IsOutOfRangeCheckEnabled()) {                                   \
    built_options.emplace("-DOUT_OF_RANGE_CHECK");                             \
  }

#define OUT_OF_RANGE_INIT(kernel)                                              \
  if (runtime->IsOutOfRangeCheckEnabled()) {                                   \
    oorc_flag = make_unique<Buffer>((context)->device()->allocator());         \
    RETURN_IF_ERROR((oorc_flag)->Allocate(sizeof(int)));                       \
    oorc_flag->Map(nullptr);                                                   \
    *(oorc_flag->mutable_data<int>()) = 0;                                     \
    oorc_flag->UnMap();                                                        \
    (kernel).setArg(0, *(static_cast<cl::Buffer *>(oorc_flag->buffer())));     \
  }

#define OUT_OF_RANGE_INIT__(kernel)                                            \
  if (runtime->IsOutOfRangeCheckEnabled()) {                                   \
    oorc_flag =                                                                \
        std::move(std::unique_ptr<Buffer>(new Buffer(device->allocator())));   \
    RETURN_IF_ERROR((oorc_flag)->Allocate(sizeof(int)));                       \
    oorc_flag->Map(nullptr);                                                   \
    *(oorc_flag->mutable_data<int>()) = 0;                                     \
    oorc_flag->UnMap();                                                        \
    (kernel).setArg(0, *(static_cast<cl::Buffer *>(oorc_flag->buffer())));     \
  }

#define OUT_OF_RANGE_SET_ARGS(kernel)                                          \
  if (runtime->IsOutOfRangeCheckEnabled()) {                                   \
    (kernel).setArg(idx++, *(static_cast<cl::Buffer *>(oorc_flag->buffer()))); \
  }

#define BUFF_OUT_OF_RANGE_SET_ARGS(kernel, size)                               \
  if (runtime->IsOutOfRangeCheckEnabled()) {                                   \
    (kernel).setArg(idx++, *(static_cast<cl::Buffer *>(oorc_flag->buffer()))); \
    (kernel).setArg(idx++, static_cast<int>(size));                            \
  }

#define OUT_OF_RANGE_VALIDATION                                                \
  if (runtime->IsOutOfRangeCheckEnabled()) {                                   \
    oorc_flag->Map(nullptr);                                                   \
    int *kerror_code = oorc_flag->mutable_data<int>();                         \
    CONDITIONS(*kerror_code == 0, "Kernel error code: ", *kerror_code);        \
    oorc_flag->UnMap();                                                        \
  }

#define NON_UNIFORM_WG_CONFIG                                                  \
  if (runtime->IsNonUniformWorkgroupsSupported()) {                            \
    built_options.emplace("-DNON_UNIFORM_WORK_GROUP");                         \
  }

#define SET_3D_GWS_ARGS(kernel, gws)                                           \
  (kernel).setArg(idx++, (gws)[0]);                                            \
  (kernel).setArg(idx++, (gws)[1]);                                            \
  (kernel).setArg(idx++, (gws)[2]);

#define SET_2D_GWS_ARGS(kernel, gws)                                           \
  (kernel).setArg(idx++, (gws)[0]);                                            \
  (kernel).setArg(idx++, (gws)[1]);

#define SET_1D_GWS_ARGS(kernel, gws) (kernel).setArg(idx++, (gws));

// Max execution time of OpenCL kernel for tuning to prevent UI stuck.
const float kMaxKernelExecTime = 1000.0; // microseconds

// Base GPU cache size used for computing local work group size.
const int32_t kBaseGPUMemCacheSize = 16384;

std::vector<index_t> FormatBufferShape(const std::vector<index_t> &buffer_shape,
                                       const OpenCLBufferType type);

std::vector<index_t> FormatBertShape(const std::vector<index_t> &buffer_shape,
                                     const OpenCLBufferType type);

// CPU data type to OpenCL command data type
std::string DtToCLCMDDt(const DataType dt);

// CPU data type to upward compatible OpenCL command data type
// e.g. half -> float
std::string DtToUpCompatibleCLCMDDt(const DataType dt);

// CPU data type to OpenCL data type
std::string DtToCLDt(const DataType dt);

// CPU data type to upward compatible OpenCL data type
// e.g. half -> float
std::string DtToUpCompatibleCLDt(const DataType dt);

// CPU data type to OpenCL condition data type used in select
// e.g. half -> float
std::string DtToCLCondDt(const DataType dt);

// Tuning or Run OpenCL kernel with 3D work group size
VanState TuningOrRun3DKernel(OpenCLRuntime *runtime,
                             const cl::Kernel &kernel,
                             const std::string tuning_key,
                             const uint32_t *gws,
                             const std::vector<uint32_t> &lws,
                             StatsFuture *future);

// Tuning or Run OpenCL kernel with 2D work group size
VanState TuningOrRun2DKernel(OpenCLRuntime *runtime,
                             const cl::Kernel &kernel,
                             const std::string tuning_key,
                             const uint32_t *gws,
                             const std::vector<uint32_t> &lws,
                             StatsFuture *future);

// Check whether limit OpenCL kernel time flag open.
inline bool LimitKernelTime() {
  const char *flag = getenv("DEEPVAN_LIMIT_OPENCL_KERNEL_TIME");
  return flag != nullptr && strlen(flag) == 1 && flag[0] == '1';
}

template <typename T>
bool IsVecEqual(const std::vector<T> &input0, const std::vector<T> &input1) {
  return ((input0.size() == input1.size()) &&
          (std::equal(input0.begin(), input0.end(), input1.begin())));
}

template <typename T>
void AppendToStream(std::stringstream *ss, const std::string &delimiter, T v) {
  UNUSED_VARIABLE(delimiter);
  (*ss) << v;
}

template <typename T, typename... Args>
void AppendToStream(std::stringstream *ss,
                    const std::string &delimiter,
                    T first,
                    Args... args) {
  (*ss) << first << delimiter;
  AppendToStream(ss, delimiter, args...);
}

template <typename... Args>
std::string Concat(Args... args) {
  std::stringstream ss;
  AppendToStream(&ss, "_", args...);
  return ss.str();
}

std::vector<uint32_t> Default2DLocalWS(OpenCLRuntime *runtime,
                                       const uint32_t *gws,
                                       const uint32_t kwg_size);

std::vector<uint32_t> Default3DLocalWS(OpenCLRuntime *runtime,
                                       const uint32_t *gws,
                                       const uint32_t kwg_size);

VanState Run1DKernel(OpenCLRuntime *runtime,
                     const cl::Kernel &kernel,
                     const uint32_t gws,
                     const uint32_t lws,
                     StatsFuture *future);

VanState Run2DKernel(OpenCLRuntime *runtime,
                     const cl::Kernel &kernel,
                     const uint32_t *gws,
                     const std::vector<uint32_t> &lws,
                     StatsFuture *future);

VanState Run3DKernel(OpenCLRuntime *runtime,
                     const cl::Kernel &kernel,
                     const uint32_t *gws,
                     const std::vector<uint32_t> &lws,
                     StatsFuture *future);

} // namespace deepvan
#endif // DEEPVAN_BACKEND_OPENCL_HELPER_H_
