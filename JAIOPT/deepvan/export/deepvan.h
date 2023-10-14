#ifndef DEEPVAN_EXPORT_DEEPVAN_H_
#define DEEPVAN_EXPORT_DEEPVAN_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef DEEPVAN_API
#define DEEPVAN_API __attribute__((visibility("default")))
#endif

namespace deepvan {
class NetProto;

enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3, HTA = 4 };

enum DataFormat {
  DF_NONE = 0,
  NHWC = 1,
  NCHW = 2,
  HWOI = 100,
  OIHW = 101,
  HWIO = 102,
  OHWI = 103
};

enum GPUPerfHint {
  PERF_DEFAULT = 0,
  PERF_LOW = 1,
  PERF_NORMAL = 2,
  PERF_HIGH = 3
};

enum GPUPriorityHint {
  PRIORITY_DEFAULT = 0,
  PRIORITY_LOW = 1,
  PRIORITY_NORMAL = 2,
  PRIORITY_HIGH = 3
};

// AFFINITY_NONE: initiate 'num_threads_hint' threads with no affinity
// scheduled.
// If 'num_threads_hint' is -1 or greater than number of available cores,
// 'num_threads_hint' will be reset to number of available cores.
// AFFINITY_BIG_ONLY: all available big cores are used, and number of threads
// is equal to numbers of available big cores.
// AFFINITY_LITTLE_ONLY: all available little cores are used, and number of
// threads is equal to numbers of available little cores.
// AFFINITY_PERFORMANCE_CORE_FIRST: initiate 'num_threads_hint' threads on different
// cores with top-num_threads_hint frequencies.
// If 'num_threads_hint' is -1 or greater than number of available cores,
// 'num_threads_hint' will be reset to number of available cores.
// AFFINITY_EFFICIENT_CORE_FIRST: initiate 'num_threads_hint' threads on different
// cores with bottom-num_threads_hint frequencies.
// If 'num_threads_hint' is -1 or greater than number of available cores,
// 'num_threads_hint' will be reset to number of available cores.
// AFFINITY_PERFORMANCE: initiate threads on meddle (or big if there are no middle core) cores. 
// it will spawn n threads, where n is the biggest power-of-2 number which ls less than the 
// number of cores in the cluster
// AFFINITY_BALANCE: initiate threads on meddle (or big if there are no middle core) cores. only one thread 
// is used, but it can run on any core in the cluster
// AFFINITY_POWER_SAVE: initiate threads on all cores. It will spawn n threads, where n is the 
// biggest power-of-2 number which ls less than the number of cores in the cluster

enum CPUAffinityPolicy {
  AFFINITY_NONE = 0,
  AFFINITY_BIG_ONLY = 1,
  AFFINITY_LITTLE_ONLY = 2,
  AFFINITY_PERFORMANCE_CORE_FIRST = 3,
  AFFINITY_EFFICIENT_CORE_FIRST = 4,
  AFFINITY_PERFORMANCE = 5,
  AFFINITY_BALANCE = 6,
  AFFINITY_POWER_SAVE = 7,
  // not used by users
  AFFINITY_CUSTOME_CLUSTER = 1024, // used internally only
  AFFINITY_CUSTOME_CPUIDS = 1025, // used internally only
};

enum SchedulePolicy {
  SCHED_DEFAULT,    // the runtime system will derive a default value according to affinity settings
  SCHED_STATIC,     // corresponds to omp_sched_static
  SCHED_DYNAMIC,    // corresponds to omp_sched_dynamic
  SCHED_GUIDED,     // corresponds to omp_sched_guided
  SCHED_AUTO,       // corresponds to omp_sched_auto
};

struct CallStats {
  int64_t start_micros;
  int64_t end_micros;
};

struct ConvPoolArgs {
  std::vector<int> strides;
  int padding_type;
  std::vector<int> paddings;
  std::vector<int> dilations;
  std::vector<int64_t> kernels;
};

struct OperatorStats {
public:
  OperatorStats(std::string name, std::string t,
                std::vector<std::vector<int64_t>> s, ConvPoolArgs a,
                CallStats c, int64_t m) {
    operator_name = name;
    type = t;
    output_shape = s;
    args = a;
    stats = c;
    macs = m;
  }
  std::string operator_name;
  std::string type;
  std::vector<std::vector<int64_t>> output_shape;
  ConvPoolArgs args;
  CallStats stats;
  int64_t macs = -1;
};

struct NetworkStats {
public:
  NetworkStats(long time): time_ms(time){}
  long time_ms;
};

class RunMetadata {
public:
  std::vector<OperatorStats> op_stats;
  std::vector<NetworkStats> net_stats;
};

/// Consistent with Android NNAPI
struct PerformanceInfo {
  // Time of executing some workload(millisecond).
  // negative value for unsupported.
  float exec_time;
};

struct Capability {
  // Performance of running with float32 data type
  // run time of the workload for CPU device,
  // ratio of run time to execute same workload compared to the time the CPU
  // execute same workload.
  PerformanceInfo float32_performance;

  // Performance of running with quantized-8 data type
  // ratio compared with float32_performance
  PerformanceInfo quantized8_performance;

  // support or not
  bool supported;
};

struct CPUCluster {
    size_t id;
    size_t num_cores;
    uint64_t freq;
    std::vector<size_t> cpu_ids;
    CPUCluster(size_t id, size_t cores, uint64_t freq, std::vector<size_t> &cpus): 
                id (id), num_cores(cores), freq(freq) {
      cpu_ids.assign(cpus.begin(), cpus.end());
    }
};

/// Get Devices Capacity
///
/// The float32_performance of CPU and GPU is tested using the workload of
/// first 8 layer of mobilenet-v2 which contain Conv(1x1, 3x3),
/// DepthwiseConv(3x3) and ElementWise Ops.
/// The quantized8_performance is just a arbitrary value tested
/// using mobilenet-v2 offline
/// Actually, It's hard to test the precise performance, the result could be
/// more accurate when your model is like with mobilenet-v2, otherwise the
/// value is just a reference.
///
/// \return capability of the device
DEEPVAN_API Capability GetCapability(DeviceType device_type,
                                     float cpu_float32_exec_time = 1.f);

class VanState {
public:
  enum Code {
    SUCCEED = 0,
    INVALID_ARGS = 1,
    OUT_OF_RESOURCES = 2,
    UNSUPPORTED = 3,
    RUNTIME_ERROR = 4,
  };

public:
  VanState();
  VanState(const Code code); // NOLINT(runtime/explicit)
  VanState(const Code code, const std::string &information);
  VanState(const VanState &);
  VanState(VanState &&);
  VanState &operator=(const VanState &);
  VanState &operator=(const VanState &&);
  ~VanState();
  Code code() const;
  std::string information() const;

  bool operator==(const VanState &other) const;
  bool operator!=(const VanState &other) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};


/// Get the number cpu cores
DEEPVAN_API size_t GetCPUCount();

/// Get the CPU clusters 
DEEPVAN_API VanState GetCPUClusters(std::vector<CPUCluster> &clusters);

/// \brief GPU context contain the status used for GPU device.
///
/// There are some data in common between different ModelExecutors using GPU,
/// use one GPUContext could avoid duplication.
///
/// Thread-safe.
/// You could use one GPUContext for multiple parallel ModelExecutors.
class GPUContext;

/// \brief GPUContext builder.
///
/// Use the GPUContextBuilder to generate GPUContext.
/// Not thread-safe
class DEEPVAN_API GPUContextBuilder {
public:
  GPUContextBuilder();
  ~GPUContextBuilder();
  GPUContextBuilder(const GPUContextBuilder &) = delete;
  GPUContextBuilder(GPUContextBuilder &&) = delete;
  GPUContextBuilder &operator=(const GPUContextBuilder &) = delete;
  GPUContextBuilder &operator=(GPUContextBuilder &&) = delete;

  /// \brief Set internal storage factory to store internal data.
  ///
  /// Now the path is used to store the built OpenCL binaries to file,
  /// which could speed up the GPU initialization and first run.
  /// If do not call this API, the initialization maybe slow for GPU.
  ///
  /// \param path  Make sure your program have Read/Write permission of the path
  /// \return
  GPUContextBuilder &SetStoragePath(const std::string &path);
  /// \brief Set paths of generated OpenCL compiled kernel binary file (not
  /// libOpenCL.so)  // NOLINT(whitespace/line_length)
  ///
  /// If you use GPU of specific soc, using OpenCL binary will speed up the
  /// initialization.  // NOLINT(whitespace/line_length) OpenCL binary is
  /// corresponding to the OpenCL Driver version, you should update the binary
  /// when OpenCL Driver changed.
  ///
  /// \param paths Deepvan will use first file found in all paths
  /// \return
  GPUContextBuilder &
  SetOpenCLBinaryPaths(const std::vector<std::string> &paths);

  /// \brief Set generated OpenCL compiled kernel binary with bytes array
  ///
  /// If you use GPU of specific soc, using OpenCL binary will speed up the
  /// initialization.  // NOLINT(whitespace/line_length) OpenCL binary is
  /// corresponding to the OpenCL Driver version, you should update the binary
  /// when OpenCL Driver changed.
  ///
  /// \param data Byte stream of OpenCL binary file
  /// \param size Size of byte stream (data)
  /// \return
  GPUContextBuilder &SetOpenCLBinary(const unsigned char *data,
                                     const size_t size);
  /// \brief Set the path of generated OpenCL parameter file
  ///
  /// If you use GPU for specific soc, the parameters is the local work group
  /// size tuned for specific SOC, which may be faster than the
  /// general parameters.
  ///
  /// \param path Make sure your program have Read/Write permission of the path
  /// \return
  GPUContextBuilder &SetOpenCLParameterPath(const std::string &path);
  /// \brief Set generated OpenCL parameter with bytes array
  ///
  /// If you use GPU for specific soc, the parameters is the local work group
  /// size tuned for specific SOC, which may be faster than the
  /// general parameters.
  ///
  /// \param data Byte stream of OpenCL parameter file
  /// \param size Size of byte stream (data)
  /// \return
  GPUContextBuilder &SetOpenCLParameter(const unsigned char *data,
                                        const size_t size);

  std::shared_ptr<GPUContext> Finalize();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class DEEPVAN_API ModelExecutionConfig {
  friend class ModelExecutor;

public:
  explicit ModelExecutionConfig(const DeviceType device_type);
  ~ModelExecutionConfig();
  ModelExecutionConfig(const ModelExecutionConfig &) = delete;
  ModelExecutionConfig(const ModelExecutionConfig &&) = delete;
  ModelExecutionConfig &operator=(const ModelExecutionConfig &) = delete;
  ModelExecutionConfig &operator=(const ModelExecutionConfig &&) = delete;

  /// \brief Set GPUContext
  ///
  /// Just use one GPUContext for multiple models run on GPU.
  /// \param context created use GPUContextBuilder
  /// \return VanState::SUCCEED for success, other for failed.
  VanState SetGPUContext(std::shared_ptr<GPUContext> context);

  /// \brief Set GPU hints, currently only supports Adreno GPU.
  ///
  /// Caution: this function may hurt performance
  /// if improper parameters provided.
  ///
  /// \param perf_hint  performance hint
  /// \param priority_hint  priority hint
  /// \return VanState::SUCCEED for success, other for failed.
  VanState SetGPUHints(GPUPerfHint perf_hint, GPUPriorityHint priority_hint);

  /// \brief Set CPU threads number and affinity policy.
  ///
  /// Caution: this function may hurt performance if improper
  /// parameters provided. When num_threads_hint is zero or negative,
  /// the function will set the threads number equaling to the number of
  /// big (AFFINITY_BIG_ONLY), little (AFFINITY_LITTLE_ONLY) or all
  /// (AFFINITY_NONE) cores according to the policy. The threads number will
  /// also be truncated to the corresponding cores number when num_threads_hint
  /// is larger than it.
  /// The OpenMP threads will be bind to (via sched_setaffinity) big cores
  /// (AFFINITY_BIG_ONLY) and little cores (AFFINITY_LITTLE_ONLY).
  ///
  /// \param num_threads_hint it is only a hint, and is only used by AFFINITY_NONE, 
  ///                         AFFINITY_BIG_ONLY, AFFINITY_LITTLE_ONLY, AFFINITY_PERFORMANCE_CORE_FIRST, 
  ///                         and AFFINITY_EFFICIENT_CORE_FIRST.
  /// \param policy one of CPUAffinityPolicy except AFFINITY_CUSTOME_CLUSTER and AFFINITY_CUSTOME_CPUIDS
  /// \return VanState::SUCCEED for success, other for failed.
  VanState SetCPUThreadPolicy(int num_threads_hint, CPUAffinityPolicy policy);
  

  /// \brief Set CPU threads number and affinity policy by specifying the cluster id.
  /// The OpenMP threads will be bind to cpu cores in the specified cluster.
  /// 
  /// \param num_threads_hint number of threads.
  /// \param cpu_cluster_id the logical cluster id of the CPU, 0 is for little core, 1 is for middle and 2 for big core.
  /// \param enable_autoset it it is set num_threads will be truncated if it is larger than the number of cores in the cluster
  /// \return VanState::SUCCEED for success, other for failed.
  VanState SetCPUThreadPolicy(int num_threads_hint, int cpu_cluster_id, bool enable_autoset = true);

  /// \brief Set CPU threads number and affinity policy by specifying the cluster id.
  /// The OpenMP threads will be bind to cpu cores in the specified cluster.
  /// 
  /// \param num_threads_hint number of threads.
  /// \param cpu_cluster_ids the logical cpu cluster ids, 0 is for little core, 1 is for middle and 2 for big core.
  /// \param enable_autoset it it is set num_threads will be truncated if it is larger than the number of cores in the cluster
  /// \param chunk_size the chunk_size to be used by openmp runtime
  /// \param omp_sched_policy one of SchedulePolicy which is used by openmp runtime
  /// \return VanState::SUCCEED for success, other for failed.
  VanState SetCPUThreadPolicy(int num_threads_hint, 
                              std::vector<size_t> &cpu_clusters, 
                              bool enable_autoset = true, 
                              int chunk_size = -1, 
                              SchedulePolicy omp_sched_policy = SCHED_DEFAULT);


  /// \brief Set CPU threads number and affinity policy by specifying the cpu ids.
  /// The OpenMP threads will be bind to specified cpu cores (represented cpu_ids) 
  /// 
  /// \param num_threads number of threads.
  /// \param omp_shced_policy one of SchedulePolicy which is used by openmp runtime.
  /// \param chunk_size the chunk_size to be used by openmp runtime.
  /// \param cpu_ids a list of cpu_ids to be used
  /// \return VanState::SUCCEED for success, other for failed.
  VanState SetCPUThreadPolicy(int num_threads, SchedulePolicy omp_sched_policy, 
                              int chunk_size, std::vector<size_t> &cpu_ids);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Deepvan input/output tensor
class DEEPVAN_API TensorWrapper {
  friend class ModelExecutor;

public:
  // shape - the shape of the tensor, with size n, if shape is unknown
  // in advance, it should be specified large enough to hold tensor of all
  // possible size.
  // data - the buffer of the tensor, must not be null with size equals
  //        shape[0] * shape[1] * ... * shape[n-1].
  //        If you want to pass a buffer which is unsuitable to use the default
  //        shared_ptr deleter (for example, the buffer is not dynamically
  //        allocated by C++, e.g. a C buffer), you can set customized deleter
  //        of shared_ptr and manage the life cycle of the buffer by yourself.
  //        For example, std::shared_ptr<float>(raw_buffer, [](float *){});
  TensorWrapper(const std::vector<int64_t> &shape, std::shared_ptr<void> data,
                const DataFormat format = DataFormat::NHWC);
  TensorWrapper();
  TensorWrapper(const TensorWrapper &other);
  TensorWrapper(const TensorWrapper &&other);
  TensorWrapper &operator=(const TensorWrapper &other);
  TensorWrapper &operator=(const TensorWrapper &&other);
  ~TensorWrapper();

  // shape will be updated to the actual output shape after running.
  const std::vector<int64_t> &shape() const;
  const std::shared_ptr<float> data() const;
  std::shared_ptr<float> data();
  template <typename T> const std::shared_ptr<T> data() const {
    return std::static_pointer_cast<T>(raw_data());
  }
  template <typename T> std::shared_ptr<T> data() {
    return std::static_pointer_cast<T>(raw_mutable_data());
  }
  DataFormat data_format() const;

private:
  std::shared_ptr<void> raw_data() const;
  std::shared_ptr<void> raw_mutable_data();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class DEEPVAN_API ModelExecutor {
public:
  explicit ModelExecutor(const ModelExecutionConfig &config);
  ~ModelExecutor();

  VanState Init(const NetProto *net_def,
                const std::vector<std::string> &input_nodes,
                const std::vector<std::string> &output_nodes,
                const unsigned char *model_data,
                const unsigned char *other_data = nullptr);

  VanState Init(const NetProto *net_def,
                const std::vector<std::string> &input_nodes,
                const std::vector<std::string> &output_nodes,
                const std::string &model_data_file,
                const std::string &other_data_file = "");

  VanState Run(const std::map<std::string, TensorWrapper> &inputs,
               std::map<std::string, TensorWrapper> *outputs);

  VanState Run(const std::map<std::string, TensorWrapper> &inputs,
               std::map<std::string, TensorWrapper> *outputs,
               RunMetadata *run_metadata);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  ModelExecutor(const ModelExecutor &) = delete;
  ModelExecutor &operator=(const ModelExecutor &) = delete;
};

/// \brief Create ModelExecutor from model graph proto and weights data
///
/// Create ModelExecutor object
///
/// \param model_graph_proto[in]: the content of model graph proto
/// \param model_graph_proto_size[in]: the size of model graph proto
/// \param model_weights_data[in]: the content of model weights data, the
///                                returned engine will refer to this buffer
///                                if CPU runtime is used. In this case, the
///                                buffer should keep alive.
/// \param model_weights_data_size[in]: the size of model weights data
/// \param input_nodes[in]: the array of input nodes' name
/// \param output_nodes[in]: the array of output nodes' name
/// \param config[in]: configurations for ModelExecutor.
/// \param engine[out]: output ModelExecutor object
/// \return VanState::SUCCEED for success,
///         VanState::INVALID_ARGS for wrong arguments,
///         VanState::OUT_OF_RESOURCES for resources is out of range.
DEEPVAN_API VanState CreateModelExecutorFromProto(
    const unsigned char *model_graph_proto, const size_t model_graph_proto_size,
    const unsigned char *model_weights_data,
    const size_t model_weights_data_size,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const ModelExecutionConfig &config, std::shared_ptr<ModelExecutor> *engine);

DEEPVAN_API VanState CreateModelExecutorFromProto(
    const unsigned char *model_graph_proto, const size_t model_graph_proto_size,
    const unsigned char *model_weights_data,
    const size_t model_weights_data_size, const unsigned char *pt_data,
    const size_t pt_data_size, const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const ModelExecutionConfig &config, std::shared_ptr<ModelExecutor> *engine);

DEEPVAN_API int InitOnGPU(const unsigned char*, 
                          const size_t,
                          const unsigned char*,
                          const size_t,
                          const std::vector<std::string> &in,
                          const std::vector<std::string> &out,
                          std::shared_ptr<ModelExecutor> *executor,
                          int num_threads, 
                          int cluster_id,
                          int hint=3);


DEEPVAN_API int InitOnCPU(const unsigned char*, 
                          const size_t,
                          const unsigned char*,
                          const size_t,
                          const std::vector<std::string> &in,
                          const std::vector<std::string> &out,
                          std::shared_ptr<ModelExecutor> *executor, 
                          int num_threads, int cluster_id);

DEEPVAN_API int RunInference(const std::string &input_name,
                             const std::vector<int64_t> &input_shape, 
                             std::shared_ptr<char> input,
                             const std::string &output_name,
                             const std::vector<int64_t> &output_shape, 
                             std::shared_ptr<char> output,
                             std::shared_ptr<deepvan::ModelExecutor> engine);

DEEPVAN_API int ReleaseExecutor(std::shared_ptr<deepvan::ModelExecutor>);

#ifdef MEMPROF_SUPPORT
DEEPVAN_API long getCPUMemoryUsage(std::shared_ptr<deepvan::ModelExecutor> engine);
DEEPVAN_API long getGPUMemoryUsage(std::shared_ptr<deepvan::ModelExecutor> engine);
#endif
} // namespace deepvan

#endif // DEEPVAN_EXPORT_DEEPVAN_H_
