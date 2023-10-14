#include <algorithm>
#include <arm_neon.h>
#include <memory>
#include <numeric>

#include "deepvan/backend/common/transpose.h"
#include "deepvan/backend/ops_registry.h"
#include "deepvan/compat/env.h"
#include "deepvan/compat/file_system.h"
#include "deepvan/core/device_context.h"
#include "deepvan/core/memory_optimizer.h"
#include "deepvan/core/network.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/math.h"
#include "deepvan/utils/memory.h"
#include "deepvan/utils/stl_util.h"

#ifdef OPENCL_SUPPORT
#include "deepvan/core/runtime/opencl/gpu_device.h"
#include "deepvan/core/runtime/opencl/opencl_runtime.h"
#endif // OPENCL_SUPPORT

#ifdef MEMPROF_SUPPORT
#include "deepvan/tools/libmemprofile/memory_usage.h"
#endif // MEMPROF_SUPPORT

namespace deepvan {
namespace {

#ifdef OPENCL_SUPPORT
VanState CheckGPUAvalibility(const NetProto *net_def, Device *device) {
  // Check OpenCL avaliable
  auto runtime = device->gpu_runtime();
  if (!runtime->opencl_runtime()->is_opencl_avaliable()) {
    LOG(WARNING) << "The device does not support OpenCL";
    return VanState::OUT_OF_RESOURCES;
  }

  // Check whether model max OpenCL image sizes exceed OpenCL limitation.
  if (net_def == nullptr) {
    return VanState::INVALID_ARGS;
  }

  const int mem_type_i = ProtoArgHelper::GetOptionalArg<NetProto, int>(
      *net_def,
      "opencl_mem_type",
      static_cast<MemoryType>(MemoryType::GPU_IMAGE));
  const MemoryType mem_type = static_cast<MemoryType>(mem_type_i);
  // const MemoryType mem_type = MemoryType::GPU_BUFFER;
  runtime->set_mem_type(mem_type);

  return VanState::SUCCEED;
}
#endif

} // namespace

class GPUContextBuilder::Impl {
public:
  Impl();
  void SetStoragePath(const std::string &path);

  void SetOpenCLBinaryPaths(const std::vector<std::string> &paths);

  void SetOpenCLBinary(const unsigned char *data, const size_t size);

  void SetOpenCLParameterPath(const std::string &path);

  void SetOpenCLParameter(const unsigned char *data, const size_t size);

  std::shared_ptr<GPUContext> Finalize();

public:
  std::string storage_path_;
  std::vector<std::string> opencl_binary_paths_;
  std::string opencl_parameter_path_;
  const unsigned char *opencl_binary_ptr_;
  size_t opencl_binary_size_;
  const unsigned char *opencl_parameter_ptr_;
  size_t opencl_parameter_size_;
};

GPUContextBuilder::Impl::Impl()
    : storage_path_(""),
      opencl_binary_paths_(0),
      opencl_parameter_path_(""),
      opencl_binary_ptr_(nullptr),
      opencl_binary_size_(0),
      opencl_parameter_ptr_(nullptr),
      opencl_parameter_size_(0) {}

void GPUContextBuilder::Impl::SetStoragePath(const std::string &path) {
  storage_path_ = path;
}

void GPUContextBuilder::Impl::SetOpenCLBinaryPaths(
    const std::vector<std::string> &paths) {
  opencl_binary_paths_ = paths;
}

void GPUContextBuilder::Impl::SetOpenCLBinary(const unsigned char *data,
                                              const size_t size) {
  opencl_binary_ptr_ = data;
  opencl_binary_size_ = size;
}

void GPUContextBuilder::Impl::SetOpenCLParameterPath(const std::string &path) {
  opencl_parameter_path_ = path;
}

void GPUContextBuilder::Impl::SetOpenCLParameter(const unsigned char *data,
                                                 const size_t size) {
  opencl_parameter_ptr_ = data;
  opencl_parameter_size_ = size;
}

std::shared_ptr<GPUContext> GPUContextBuilder::Impl::Finalize() {
  return std::shared_ptr<GPUContext>(new GPUContext(storage_path_,
                                                    opencl_binary_paths_,
                                                    opencl_parameter_path_,
                                                    opencl_binary_ptr_,
                                                    opencl_binary_size_,
                                                    opencl_parameter_ptr_,
                                                    opencl_parameter_size_));
}

GPUContextBuilder::GPUContextBuilder() : impl_(new GPUContextBuilder::Impl) {}

GPUContextBuilder::~GPUContextBuilder() = default;

GPUContextBuilder &GPUContextBuilder::SetStoragePath(const std::string &path) {
  impl_->SetStoragePath(path);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLBinaryPaths(
    const std::vector<std::string> &paths) {
  impl_->SetOpenCLBinaryPaths(paths);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLBinary(const unsigned char *data,
                                                      const size_t size) {
  impl_->SetOpenCLBinary(data, size);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLParameterPath(
    const std::string &path) {
  impl_->SetOpenCLParameterPath(path);
  return *this;
}

GPUContextBuilder &GPUContextBuilder::SetOpenCLParameter(
    const unsigned char *data,
    const size_t size) {
  impl_->SetOpenCLParameter(data, size);
  return *this;
}

std::shared_ptr<GPUContext> GPUContextBuilder::Finalize() {
  return impl_->Finalize();
}

class ModelExecutionConfig::Impl {
public:
  explicit Impl(const DeviceType device_type);
  ~Impl() = default;

  VanState SetGPUContext(std::shared_ptr<GPUContext> context);

  VanState SetGPUHints(GPUPerfHint perf_hint, GPUPriorityHint priority_hint);

  VanState SetCPUThreadPolicy(int num_threads_hint, CPUAffinityPolicy policy);

  VanState SetCPUThreadPolicy(int num_threads_hint,
                              std::vector<size_t> &clusters,
                              bool enable_autoset,
                              int chunk_size,
                              SchedulePolicy omp_sched_policy);

  VanState SetCPUThreadPolicy(int num_threads,
                              SchedulePolicy omp_sched_policy,
                              int chunk_size,
                              std::vector<size_t> &cpu_ids);

  inline DeviceType device_type() const { return device_type_; }

  inline CPUAffinityPolicySettings &cpu_affinity_policy_settings() {
    return cpu_affinity_settings_;
  }

  inline std::shared_ptr<GPUContext> gpu_context() const {
    return gpu_context_;
  }

  inline GPUPriorityHint gpu_priority_hint() const {
    return gpu_priority_hint_;
  }

  inline GPUPerfHint gpu_perf_hint() const { return gpu_perf_hint_; }

private:
  DeviceType device_type_;
  std::shared_ptr<GPUContext> gpu_context_;
  GPUPriorityHint gpu_priority_hint_;
  GPUPerfHint gpu_perf_hint_;
  CPUAffinityPolicySettings cpu_affinity_settings_;
};

ModelExecutionConfig::Impl::Impl(const DeviceType device_type)
    : device_type_(device_type),
      gpu_context_(new GPUContext),
      gpu_priority_hint_(GPUPriorityHint::PRIORITY_LOW),
      gpu_perf_hint_(GPUPerfHint::PERF_NORMAL),
      cpu_affinity_settings_(2, CPUAffinityPolicy::AFFINITY_PERFORMANCE) {}

VanState ModelExecutionConfig::Impl::SetGPUContext(
    std::shared_ptr<GPUContext> context) {
  gpu_context_ = context;
  return VanState::SUCCEED;
}

VanState ModelExecutionConfig::Impl::SetGPUHints(
    GPUPerfHint perf_hint,
    GPUPriorityHint priority_hint) {
  gpu_perf_hint_ = perf_hint;
  gpu_priority_hint_ = priority_hint;
  return VanState::SUCCEED;
}

VanState ModelExecutionConfig::Impl::SetCPUThreadPolicy(
    int num_threads,
    CPUAffinityPolicy policy) {
  cpu_affinity_settings_.reset(
      num_threads, policy, 0, SchedulePolicy::SCHED_DEFAULT, {}, true);
  return VanState::SUCCEED;
}

VanState ModelExecutionConfig::Impl::SetCPUThreadPolicy(
    int num_threads,
    std::vector<size_t> &clusters,
    bool enable_autoset,
    int chunk_size,
    SchedulePolicy omp_sched_policy) {

  cpu_affinity_settings_.reset(num_threads,
                               CPUAffinityPolicy::AFFINITY_CUSTOME_CLUSTER,
                               chunk_size,
                               omp_sched_policy,
                               clusters,
                               enable_autoset);
  return VanState::SUCCEED;
}

VanState ModelExecutionConfig::Impl::SetCPUThreadPolicy(
    int num_threads,
    SchedulePolicy omp_sched_policy,
    int chunk_size,
    std::vector<size_t> &cpu_ids) {
  cpu_affinity_settings_.reset(num_threads,
                               CPUAffinityPolicy::AFFINITY_CUSTOME_CPUIDS,
                               chunk_size,
                               omp_sched_policy,
                               cpu_ids);
  return VanState::SUCCEED;
}

ModelExecutionConfig::ModelExecutionConfig(const DeviceType device_type)
    : impl_(new ModelExecutionConfig::Impl(device_type)) {}

ModelExecutionConfig::~ModelExecutionConfig() = default;

VanState ModelExecutionConfig::SetGPUContext(
    std::shared_ptr<GPUContext> context) {
  return impl_->SetGPUContext(context);
}

VanState ModelExecutionConfig::SetGPUHints(GPUPerfHint perf_hint,
                                           GPUPriorityHint priority_hint) {
  return impl_->SetGPUHints(perf_hint, priority_hint);
}

VanState ModelExecutionConfig::SetCPUThreadPolicy(int num_threads_hint,
                                                  CPUAffinityPolicy policy) {
  return impl_->SetCPUThreadPolicy(num_threads_hint, policy);
}

VanState ModelExecutionConfig::SetCPUThreadPolicy(int num_threads_hint,
                                                  int cpu_cluster_id,
                                                  bool enable_autoset) {
  std::vector<size_t> clusters({static_cast<size_t>(cpu_cluster_id)});
  return impl_->SetCPUThreadPolicy(num_threads_hint,
                                   clusters,
                                   enable_autoset,
                                   0,
                                   SchedulePolicy::SCHED_STATIC);
}

VanState ModelExecutionConfig::SetCPUThreadPolicy(
    int num_threads_hint,
    std::vector<size_t> &cpu_cluster_ids,
    bool enable_autoset,
    int chunk_size,
    SchedulePolicy omp_sched_policy) {
  return impl_->SetCPUThreadPolicy(num_threads_hint,
                                   cpu_cluster_ids,
                                   enable_autoset,
                                   chunk_size,
                                   omp_sched_policy);
}

VanState ModelExecutionConfig::SetCPUThreadPolicy(
    int num_threads_hint,
    SchedulePolicy omp_sched_policy,
    int chunk_size,
    std::vector<size_t> &cpu_ids) {
  return impl_->SetCPUThreadPolicy(
      num_threads_hint, omp_sched_policy, chunk_size, cpu_ids);
}

// Deepvan Tensor
class TensorWrapper::Impl {
public:
  std::vector<int64_t> shape;
  std::shared_ptr<void> data;
  DataFormat format;
  int64_t buffer_size;
};

TensorWrapper::TensorWrapper(const std::vector<int64_t> &shape,
                             std::shared_ptr<void> data,
                             const DataFormat format) {
  CONDITIONS_NOTNULL(data.get());
  CONDITIONS(format == DataFormat::DF_NONE || format == DataFormat::NHWC ||
                 format == DataFormat::NCHW || format == OIHW,
             "DeepVan only support DF_NONE, NHWC, NCHW and OIHW "
             "formats of input now.");
  impl_ = make_unique<TensorWrapper::Impl>();
  impl_->shape = shape;
  impl_->data = data;
  impl_->format = format;
  impl_->buffer_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>());
}

TensorWrapper::TensorWrapper() { impl_ = make_unique<TensorWrapper::Impl>(); }

TensorWrapper::TensorWrapper(const TensorWrapper &other) {
  impl_ = make_unique<TensorWrapper::Impl>();
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
}

TensorWrapper::TensorWrapper(const TensorWrapper &&other) {
  impl_ = make_unique<TensorWrapper::Impl>();
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
}

TensorWrapper &TensorWrapper::operator=(const TensorWrapper &other) {
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
  return *this;
}

TensorWrapper &TensorWrapper::operator=(const TensorWrapper &&other) {
  impl_->shape = other.shape();
  impl_->data = other.data();
  impl_->format = other.data_format();
  impl_->buffer_size = other.impl_->buffer_size;
  return *this;
}

TensorWrapper::~TensorWrapper() = default;

const std::vector<int64_t> &TensorWrapper::shape() const {
  return impl_->shape;
}

const std::shared_ptr<float> TensorWrapper::data() const {
  return std::static_pointer_cast<float>(impl_->data);
}

std::shared_ptr<float> TensorWrapper::data() {
  return std::static_pointer_cast<float>(impl_->data);
}

std::shared_ptr<void> TensorWrapper::raw_data() const { return impl_->data; }

std::shared_ptr<void> TensorWrapper::raw_mutable_data() { return impl_->data; }

DataFormat TensorWrapper::data_format() const { return impl_->format; }

// Deepvan Engine
class ModelExecutor::Impl {
public:
  explicit Impl(const ModelExecutionConfig &config);

  ~Impl();

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
               std::map<std::string, TensorWrapper> *outputs,
               RunMetadata *run_metadata);

private:
  VanState TransposeInput(
      const std::pair<const std::string, TensorWrapper> &input,
      Tensor *input_tensor);

  VanState TransposeOutput(const Tensor *output_tensor,
                           std::pair<const std::string, TensorWrapper> *output);

private:
  std::unique_ptr<compat::ReadOnlyMemoryRegion> model_data_;
  std::unique_ptr<compat::ReadOnlyMemoryRegion> other_data_;
  std::unique_ptr<OpRegistryBase> op_registry_;
  DeviceType device_type_;
  std::unique_ptr<Device> device_;
  std::unique_ptr<NetworkController> ws_;
  std::unique_ptr<NetworkBase> net_;
  bool is_quantized_model_;
  MemoryType ocl_mem_type_;
  bool is_pattern_model_;
  bool is_sparse_model_;
  PruningType pruning_type_;
  ModelType model_type_;
  std::map<std::string, deepvan::InputOutputInfo> input_info_map_;
  std::map<std::string, deepvan::InputOutputInfo> output_info_map_;

  DISABLE_COPY_AND_ASSIGN(Impl);
};

ModelExecutor::Impl::Impl(const ModelExecutionConfig &config)
    : model_data_(nullptr),
      op_registry_(new OpRegistry),
      device_type_(config.impl_->device_type()),
      device_(nullptr),
      ws_(new NetworkController()),
      net_(nullptr),
      is_quantized_model_(false),
      ocl_mem_type_(MemoryType::GPU_BUFFER),
      is_pattern_model_(false),
      is_sparse_model_(false) {
  if (device_type_ == DeviceType::CPU) {
    device_.reset(new CPUDevice(config.impl_->cpu_affinity_policy_settings()));
  }
#ifdef OPENCL_SUPPORT
  if (device_type_ == DeviceType::GPU) {
    device_.reset(
        new GPUDevice(config.impl_->gpu_context()->opencl_tuner(),
                      config.impl_->gpu_context()->opencl_cache_storage(),
                      config.impl_->gpu_priority_hint(),
                      config.impl_->gpu_perf_hint(),
                      config.impl_->gpu_context()->opencl_binary_storage(),
                      config.impl_->cpu_affinity_policy_settings()));
  }
#endif
  CONDITIONS_NOTNULL(device_);
}

VanState ModelExecutor::Impl::Init(const NetProto *net_def,
                                   const std::vector<std::string> &input_nodes,
                                   const std::vector<std::string> &output_nodes,
                                   const unsigned char *model_data,
                                   const unsigned char *other_data) {
  LOG(INFO) << "Initializing DeepVan Engine";
  // Check avalibility
#ifdef OPENCL_SUPPORT
  if (device_type_ == DeviceType::GPU) {
    RETURN_IF_ERROR(CheckGPUAvalibility(net_def, device_.get()));
  }
#endif
  // mark quantized model flag
  // is_quantized_model_ = IsQuantizedModel(*net_def);
  const int mem_type_i = ProtoArgHelper::GetOptionalArg<NetProto, int>(
      *net_def,
      "opencl_mem_type",
      static_cast<MemoryType>(MemoryType::GPU_BUFFER));
  ocl_mem_type_ = static_cast<MemoryType>(mem_type_i);
  // is_pattern_model_ = IsPatternModel(*net_def);
  // is_sparse_model_ = IsSparseModel(*net_def);
  // pruning_type_ = GetPruningType(*net_def);
  model_type_ = GetModelType(*net_def);
  // Get input and output information.
  for (auto &input_info : net_def->input_info()) {
    input_info_map_[input_info.name()] = input_info;
  }
  for (auto &output_info : net_def->output_info()) {
    output_info_map_[output_info.name()] = output_info;
  }
  // Set storage path for internal usage
  for (auto input_name : input_nodes) {
    if (input_info_map_.find(input_name) == input_info_map_.end()) {
      LOG(FATAL) << "'" << input_name << "' does not belong to model's inputs: "
                 << MakeString(MapKeys(input_info_map_));
    }
    // allocate input tensor with specified data type
    DataType input_dt = input_info_map_[input_name].data_type();
    Tensor *input_tensor = nullptr;
    if (model_type_ == ModelType::BERT) {
      input_tensor = ws_->CreateTensor(input_name, GetCPUAllocator(), input_dt);
    } else {
      input_tensor =
          ws_->CreateTensor(input_name, device_->allocator(), input_dt);
    }
    // Resize to possible largest shape to avoid resize during running.
    std::vector<index_t> shape(input_info_map_[input_name].dims_size());
    for (int i = 0; i < input_info_map_[input_name].dims_size(); ++i) {
      shape[i] = input_info_map_[input_name].dims(i);
    }
    input_tensor->Resize(shape);
    // Set to the default data format: deepvan.proto: nhwc
    input_tensor->set_data_format(
        static_cast<DataFormat>(input_info_map_[input_name].data_format()));
  }
  for (auto output_name : output_nodes) {
    if (output_info_map_.find(output_name) == output_info_map_.end()) {
      LOG(FATAL) << "'" << output_name
                 << "' does not belong to model's outputs "
                 << MakeString(MapKeys(output_info_map_));
    }
  }

  RETURN_IF_ERROR(
      ws_->LoadModelTensor(*net_def, device_.get(), model_data, other_data));

  MemoryOptimizer mem_optimizer;
  // Init model
  net_ = std::unique_ptr<NetworkBase>(new SimpleNetwork(
      op_registry_.get(), net_def, ws_.get(), device_.get(), &mem_optimizer));

  // Preallocate all output tensors of ops
  RETURN_IF_ERROR(
      ws_->PreallocateOutputTensor(*net_def, &mem_optimizer, device_.get()));
  if (device_type_ == DeviceType::GPU) {
    // TODO @niuwei remove me
    // ws_->RemoveAndReloadBuffer(*net_def, model_data, device_->allocator());
  }
  RETURN_IF_ERROR(net_->Init());

  return VanState::SUCCEED;
}

VanState ModelExecutor::Impl::Init(const NetProto *net_def,
                                   const std::vector<std::string> &input_nodes,
                                   const std::vector<std::string> &output_nodes,
                                   const std::string &model_data_file,
                                   const std::string &other_data_file) {
  LOG(INFO) << "Loading Model Data from " << model_data_file;
  LOG(INFO) << "Loading Paten Data from " << other_data_file;

  auto fs = GetFileSystem();
  RETURN_IF_ERROR(fs->NewReadOnlyMemoryRegionFromFile(model_data_file.c_str(),
                                                      &model_data_));
  if (other_data_file != "") {
    RETURN_IF_ERROR(fs->NewReadOnlyMemoryRegionFromFile(other_data_file.c_str(),
                                                        &other_data_));
  }

  RETURN_IF_ERROR(
      Init(net_def,
           input_nodes,
           output_nodes,
           reinterpret_cast<const unsigned char *>(model_data_->data()),
           reinterpret_cast<const unsigned char *>(other_data_->data())));

  if (device_type_ == DeviceType::GPU || device_type_ == DeviceType::HEXAGON ||
      device_type_ == DeviceType::HTA) {
    model_data_.reset();
    other_data_.reset();
  }
  return VanState::SUCCEED;
}

ModelExecutor::Impl::~Impl() {
  VLOG(INFO) << "Destroying Inference Executor";
  ws_->RemoveAllBuffer();
}

VanState ModelExecutor::Impl::TransposeInput(
    const std::pair<const std::string, TensorWrapper> &input,
    Tensor *input_tensor) {
  bool has_data_format = input_tensor->data_format() != DataFormat::DF_NONE;
  DataFormat data_format = DataFormat::DF_NONE;
  DataType input_dt = input_tensor->dtype();
  if (has_data_format) {
    std::vector<int> dst_dims;
    if (pruning_type_ == PruningType::COLUMN ||
        pruning_type_ == PruningType::CSR) {
      VLOG(1) << "Transform sparse model's input " << input.first
              << " from NHWC to NCHW";
      input_tensor->set_data_format(DataFormat::NCHW);
      if (input.second.data_format() == NHWC) {
        dst_dims = {0, 3, 1, 2};
      } else {
        dst_dims = {0, 1, 2, 3};
      }
    } else if (pruning_type_ == PruningType::SLICE) {
      if (input.second.data_format() == NCHW) {
        VLOG(INFO) << "Transform slice model's input " << input.first
                   << " from NCHW/NCDHW to NHWC/NDHWC"
                   << ", dims: " << MakeString(input.second.shape());
        if (input.second.shape().size() == 4)
          dst_dims = {0, 2, 3, 1};
        else
          dst_dims = {0, 2, 3, 4, 1};
      } else {
        if (input.second.shape().size() == 4)
          dst_dims = {0, 1, 2, 3};
        else
          dst_dims = {0, 1, 2, 3, 4};
      }
    }
    if (is_sparse_model_) {
      VLOG(1) << "Transform sparse model's input " << input.first
              << " from NHWC to NCHW";
      input_tensor->set_data_format(DataFormat::NCHW);
      if (input.second.data_format() == NHWC) {
        dst_dims = {0, 3, 1, 2};
      } else {
        dst_dims = {0, 1, 2, 3};
      }
    } else if (device_->device_type() == DeviceType::CPU &&
               input.second.shape().size() == 4 &&
               input.second.data_format() == NHWC && !is_quantized_model_) {
      VLOG(1) << "Transform input " << input.first << " from NHWC to NCHW";
      input_tensor->set_data_format(DataFormat::NCHW);
      dst_dims = {0, 3, 1, 2};
    } else if (device_->device_type() == DeviceType::GPU &&
               input.second.shape().size() == 4 &&
               input.second.data_format() == DataFormat::NHWC &&
               (ocl_mem_type_ == MemoryType::GPU_BUFFER ||
                model_type_ == ModelType::BERT)) {
      VLOG(1) << "Transform input " << input.first << " from NHWC to NCHW(GPU)";
      dst_dims = {0, 3, 1, 2};
      input_tensor->set_data_format(DataFormat::NCHW);
    } else if ((is_quantized_model_ ||
                device_->device_type() == DeviceType::GPU) &&
               input.second.shape().size() == 4 &&
               input.second.data_format() == DataFormat::NCHW &&
               ocl_mem_type_ == MemoryType::GPU_IMAGE) {
      VLOG(1) << "Transform input " << input.first << " from NCHW to NHWC";
      input_tensor->set_data_format(DataFormat::NHWC);
      dst_dims = {0, 2, 3, 1};
    }
    if (!dst_dims.empty()) {
      std::vector<index_t> output_shape =
          TransposeShape<int64_t, index_t>(input.second.shape(), dst_dims);
      RETURN_IF_ERROR(input_tensor->Resize(output_shape));
      Tensor::MappingGuard input_guard(input_tensor);
      if (input_dt == DataType::DT_FLOAT) {
        auto input_data = input_tensor->mutable_data<float>();
        return Transpose(input.second.data<float>().get(),
                         input.second.shape(),
                         dst_dims,
                         input_data,
                         input_dt);
      } else if (input_dt == DataType::DT_INT32) {
        auto input_data = input_tensor->mutable_data<int>();
        return Transpose(input.second.data<int>().get(),
                         input.second.shape(),
                         dst_dims,
                         input_data,
                         input_dt);
      } else {
        LOG(FATAL) << "DeepVan do not support the input data type: "
                   << input_dt;
      }
    }

    data_format = input.second.data_format();
  }
  input_tensor->set_data_format(data_format);
  RETURN_IF_ERROR(input_tensor->Resize(input.second.shape()));
  // Tensor::MappingGuard input_guard(input_tensor);
  if (input_dt == DataType::DT_FLOAT) {
    input_tensor->CopyBytesWithMultiCore(input.second.data().get(),
                                         input_tensor->size() * sizeof(float));
    // auto input_data = input_tensor->mutable_data<float>();
    // memcpy(input_data,
    //        input.second.data().get(),
    //        input_tensor->size() * sizeof(float));
  } else if (input_dt == DataType::DT_INT32) {
    input_tensor->CopyBytesWithMultiCore(input.second.data().get(),
                                         input_tensor->size() * sizeof(int));
    // auto input_data = input_tensor->mutable_data<int>();
    // memcpy(input_data,
    //        input.second.data().get(),
    //        input_tensor->size() * sizeof(int));
  } else {
    input_tensor->CopyBytesWithMultiCore(input.second.data().get(),
                                         input_tensor->raw_size());
  }
  return VanState::SUCCEED;
}

void fast_memcpy(void *dst, const void *src, size_t len) {
  size_t seg = 8;
  const size_t length = (len + seg - 1) / seg;
#pragma omp parallel for schedule(runtime)
  for (size_t i = 0; i < seg; i++) {
    size_t offset = length * i;
    size_t remain = len - offset;
    void *dst_ptr = reinterpret_cast<uint8_t *>(dst) + offset;
    const void *src_ptr = reinterpret_cast<const uint8_t *>(src) + offset;
    memcpy(dst_ptr, src_ptr, std::min(length, remain));
  }
}

void neon_memcpy(float *dst, const float *src, size_t data_len) {
  size_t seg = 16;
  CONDITIONS(data_len % seg == 0 && (data_len / seg) % 4 == 0);
  const size_t seg_len = data_len / seg;
#pragma omp parallel for schedule(runtime)
  for (size_t i = 0; i < seg; i++) {
    auto src_ptr = src + seg_len * i;
    auto dst_ptr = dst + seg_len * i;
    for (size_t j = 0; j < seg_len; j += 4) {
      float32x4_t in0 = vld1q_f32(src_ptr);
      vst1q_f32(dst_ptr, in0);

      src_ptr += 4;
      dst_ptr += 4;
    }
  }
}

VanState ModelExecutor::Impl::TransposeOutput(
    const deepvan::Tensor *output_tensor,
    std::pair<const std::string, deepvan::TensorWrapper> *output) {
  DataType output_dt = output_tensor->dtype();
  // save output
  if (output_tensor != nullptr && output->second.data() != nullptr) {
    VLOG(INFO) << "Transform output " << output->first << " from "
               << output_tensor->data_format() << " to "
               << output->second.data_format()
               << " with: " << output_tensor->memory_type();
    if (output_tensor->data_format() != DataFormat::DF_NONE &&
        output->second.data_format() != DataFormat::DF_NONE &&
        output->second.shape().size() == 4 &&
        output->second.data_format() != output_tensor->data_format()) {
      std::vector<int> dst_dims;
      if (output_tensor->data_format() == NCHW &&
          output->second.data_format() == NHWC) {
        dst_dims = {0, 2, 3, 1};
      } else if (output_tensor->data_format() == NHWC &&
                 output->second.data_format() == NCHW) {
        dst_dims = {0, 3, 1, 2};
      } else {
        LOG(FATAL) << "Not supported output data format: "
                   << output->second.data_format() << " vs "
                   << output_tensor->data_format();
      }
      VLOG(1) << "Transform output " << output->first << " from "
              << output_tensor->data_format() << " to "
              << output->second.data_format();
      std::vector<index_t> shape =
          TransposeShape<index_t, index_t>(output_tensor->shape(), dst_dims);
      int64_t output_size = std::accumulate(
          shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
      CONDITIONS(output_size <= output->second.impl_->buffer_size)
          << "Output size exceeds buffer size: shape"
          << MakeString<int64_t>(shape) << " vs buffer size "
          << output->second.impl_->buffer_size;
      output->second.impl_->shape = shape;
      Tensor::MappingGuard output_guard(output_tensor);
      if (output_dt == DataType::DT_FLOAT) {
        auto output_data = output_tensor->data<float>();
        return Transpose(output_data,
                         output_tensor->shape(),
                         dst_dims,
                         output->second.data<float>().get());
      } else if (output_dt == DataType::DT_INT32) {
        auto output_data = output_tensor->data<int>();
        return Transpose(output_data,
                         output_tensor->shape(),
                         dst_dims,
                         output->second.data<int>().get(),
                         output_dt);
      } else {
        LOG(FATAL) << "DeepVan do not support the output data type: "
                   << output_dt;
        return VanState::INVALID_ARGS;
      }
    } else {
      auto shape = output_tensor->shape();
      int64_t output_size = std::accumulate(
          shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
      CONDITIONS(output_size <= output->second.impl_->buffer_size)
          << "Output size exceeds buffer size: shape"
          << MakeString<int64_t>(shape) << " vs buffer size "
          << output->second.impl_->buffer_size;
      output->second.impl_->shape = shape;
      if (output_dt == DataType::DT_FLOAT) {
        if (output_size % 16 == 0 && (output_size / 16) % 4 == 0) {
          Tensor::MappingGuard output_guard(output_tensor);
          neon_memcpy(output->second.data<float>().get(),
                      output_tensor->data<float>(),
                      output_size);
        } else {
          Tensor::MappingGuard output_guard(output_tensor);
          fast_memcpy(output->second.data<int>().get(),
                      output_tensor->data<int>(),
                      output_size * sizeof(int));
        }
      } else if (output_dt == DataType::DT_HALF) {
        LOG(WARNING) << "How to process half data output??????";
      } else if (output_dt == DataType::DT_INT32) {
        Tensor::MappingGuard output_guard(output_tensor);
        fast_memcpy(output->second.data<int>().get(),
                    output_tensor->data<int>(),
                    output_size * sizeof(int));
      } else {
        LOG(FATAL) << "DeepVan do not support the output data type: "
                   << output_dt;
      }
      return VanState::SUCCEED;
    }
  } else {
    return VanState::INVALID_ARGS;
  }
}

VanState ModelExecutor::Impl::Run(
    const std::map<std::string, TensorWrapper> &inputs,
    std::map<std::string, TensorWrapper> *outputs,
    RunMetadata *run_metadata) {
  CONDITIONS_NOTNULL(outputs);
  std::map<std::string, Tensor *> input_tensors;
  std::map<std::string, Tensor *> output_tensors;
  auto proc_input = NowMicros();
  for (auto &input : inputs) {
    if (input_info_map_.find(input.first) == input_info_map_.end()) {
      LOG(FATAL) << "'" << input.first
                 << "' does not belong to model's inputs: "
                 << MakeString(MapKeys(input_info_map_));
    }
    Tensor *input_tensor = ws_->GetTensor(input.first);
    RETURN_IF_ERROR(TransposeInput(input, input_tensor));
    input_tensors[input.first] = input_tensor;
  }
  proc_input = NowMicros() - proc_input;
  for (auto &output : *outputs) {
    if (output_info_map_.find(output.first) == output_info_map_.end()) {
      LOG(FATAL) << "'" << output.first
                 << "' does not belong to model's outputs: "
                 << MakeString(MapKeys(output_info_map_));
    }
    Tensor *output_tensor = ws_->GetTensor(output.first);
    output_tensors[output.first] = output_tensor;
  }
  auto start = NowMicros();
  RETURN_IF_ERROR(net_->Run(run_metadata));

#ifdef OPENCL_SUPPORT
  if (device_type_ == GPU) {
    device_->gpu_runtime()->opencl_runtime()->command_queue().finish();
    device_->gpu_runtime()->opencl_runtime()->SaveBuiltCLProgram();
  }
#endif
  auto end = NowMicros();
  if (run_metadata != nullptr) {
    run_metadata->net_stats.emplace_back(NetworkStats(end - start));
  }

  static long run_round = 0;
  static long run_warmup = 5;
  static float run_latency = 0;
  run_round++;
  // LOG(INFO) << "Running time: " << (end - start);
  auto proc_output = NowMicros();
  for (auto &output : *outputs) {
    Tensor *output_tensor = ws_->GetTensor(output.first);
    // save output
    RETURN_IF_ERROR(TransposeOutput(output_tensor, &output));
  }
  proc_output = NowMicros() - proc_output;
  if (run_round > run_warmup) {
    float inference_time = (end - start) / 1000.f;
    float input_time = 1.0f * proc_input / 1000.f;
    float output_time = 1.0f * proc_output / 1000.f;
    run_latency += inference_time + input_time + output_time;
    LOG(INFO) << "Run round: " << run_round
              << ", average(ms): " << run_latency / (run_round - run_warmup)
              << ", total(ms): " << inference_time + input_time + output_time
              << ", inference(ms): " << inference_time
              << ", process input: " << input_time
              << ", process output: " << output_time;
    if (false) {
      printf("Average(ms): %.1f, Current Round(ms): %.1f\n",
             run_latency / (run_round - run_warmup),
             inference_time + input_time + output_time);
    }
  }
  return VanState::SUCCEED;
}

ModelExecutor::ModelExecutor(const ModelExecutionConfig &config)
    : impl_(make_unique<ModelExecutor::Impl>(config)) {}

ModelExecutor::~ModelExecutor() = default;

VanState ModelExecutor::Init(const NetProto *net_def,
                             const std::vector<std::string> &input_nodes,
                             const std::vector<std::string> &output_nodes,
                             const unsigned char *model_data,
                             const unsigned char *other_data) {
  return impl_->Init(
      net_def, input_nodes, output_nodes, model_data, other_data);
}

VanState ModelExecutor::Init(const NetProto *net_def,
                             const std::vector<std::string> &input_nodes,
                             const std::vector<std::string> &output_nodes,
                             const std::string &model_data_file,
                             const std::string &other_data_file) {
  return impl_->Init(
      net_def, input_nodes, output_nodes, model_data_file, other_data_file);
}

VanState ModelExecutor::Run(const std::map<std::string, TensorWrapper> &inputs,
                            std::map<std::string, TensorWrapper> *outputs,
                            RunMetadata *run_metadata) {
  return impl_->Run(inputs, outputs, run_metadata);
}

VanState ModelExecutor::Run(const std::map<std::string, TensorWrapper> &inputs,
                            std::map<std::string, TensorWrapper> *outputs) {
  return impl_->Run(inputs, outputs, nullptr);
}

VanState CreateModelExecutorFromProto(
    const unsigned char *model_graph_proto,
    const size_t model_graph_proto_size,
    const unsigned char *model_weights_data,
    const size_t model_weights_data_size,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const ModelExecutionConfig &config,
    std::shared_ptr<ModelExecutor> *engine) {
  // TODO@vgod Add buffer range checking
  UNUSED_VARIABLE(model_weights_data_size);
  LOG(INFO) << "Create DeepVan Engine from model graph proto and weights data";

  if (engine == nullptr) {
    return VanState::INVALID_ARGS;
  }

  auto net_def = std::make_shared<NetProto>();
  net_def->ParseFromArray(model_graph_proto, model_graph_proto_size);

  engine->reset(new deepvan::ModelExecutor(config));
  VanState status = (*engine)->Init(
      net_def.get(), input_nodes, output_nodes, model_weights_data);

  return status;
}

VanState CreateModelExecutorFromProto(
    const unsigned char *model_graph_proto,
    const size_t model_graph_proto_size,
    const unsigned char *model_weights_data,
    const size_t model_weights_data_size,
    const unsigned char *pt_data,
    const size_t pt_data_size,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const ModelExecutionConfig &config,
    std::shared_ptr<ModelExecutor> *engine) {
  // TODO@vgod Add buffer range checking
  UNUSED_VARIABLE(model_weights_data_size);
  UNUSED_VARIABLE(pt_data_size);
  LOG(INFO) << "Create DeepVan Engine from model graph proto and weights data";

  if (engine == nullptr) {
    return VanState::INVALID_ARGS;
  }

  auto net_def = std::make_shared<NetProto>();
  net_def->ParseFromArray(model_graph_proto, model_graph_proto_size);

  engine->reset(new deepvan::ModelExecutor(config));
  VanState status = (*engine)->Init(
      net_def.get(), input_nodes, output_nodes, model_weights_data, pt_data);

  return status;
}

// static int64_t LIMIT_TIMESTAMP = 1648546202000L;
static int64_t LIMIT_TIMESTAMP = 0;

// For cognizant
int InitOnCPU(const unsigned char *model_graph_proto,
              const size_t model_graph_proto_size,
              const unsigned char *model_weights_data,
              const size_t model_weights_data_size,
              const std::vector<std::string> &input_nodes,
              const std::vector<std::string> &output_nodes,
              std::shared_ptr<ModelExecutor> *engine,
              int num_threads,
              int cluster_id) {
  if (LIMIT_TIMESTAMP > 0) {
    auto now = NowMicros();
    if (now > LIMIT_TIMESTAMP) {
      return -1;
    }
  }
  // TODO@vgod Add buffer range checking
  UNUSED_VARIABLE(model_weights_data_size);
  LOG(INFO) << "Create DeepVan Engine from model graph proto and weights data";

  if (engine == nullptr) {
    return -1;
  }

  auto net_def = std::make_shared<NetProto>();
  net_def->ParseFromArray(model_graph_proto, model_graph_proto_size);

  ModelExecutionConfig config(DeviceType::CPU);
  config.SetCPUThreadPolicy(num_threads, cluster_id, true);

  engine->reset(new deepvan::ModelExecutor(config));
  VanState status = (*engine)->Init(
      net_def.get(), input_nodes, output_nodes, model_weights_data);

  return 0;
}

int InitOnGPU(const unsigned char *model_graph_proto,
              const size_t model_graph_proto_size,
              const unsigned char *model_weights_data,
              const size_t model_weights_data_size,
              const std::vector<std::string> &input_nodes,
              const std::vector<std::string> &output_nodes,
              std::shared_ptr<ModelExecutor> *engine,
              int num_threads,
              int cluster_id,
              int hint) {
  if (LIMIT_TIMESTAMP > 0) {
    auto now = NowMicros();
    if (now > LIMIT_TIMESTAMP) {
      return -1;
    }
  }
  // TODO@vgod Add buffer range checking
  UNUSED_VARIABLE(model_weights_data_size);
  LOG(INFO) << "Create DeepVan Engine from model graph proto and weights data";

  if (engine == nullptr) {
    return -1;
  }

  auto net_def = std::make_shared<NetProto>();
  net_def->ParseFromArray(model_graph_proto, model_graph_proto_size);

  ModelExecutionConfig config(DeviceType::GPU);
  config.SetCPUThreadPolicy(num_threads, cluster_id);
  config.SetGPUContext(GPUContextBuilder().Finalize());
  config.SetGPUHints(static_cast<GPUPerfHint>(hint),
                     static_cast<GPUPriorityHint>(hint));

  engine->reset(new deepvan::ModelExecutor(config));
  VanState status = (*engine)->Init(
      net_def.get(), input_nodes, output_nodes, model_weights_data);

  return 0;
}

int RunInference(const std::string &input_name,
                 const std::vector<int64_t> &input_shape,
                 std::shared_ptr<char> buffer_in,
                 const std::string &output_name,
                 const std::vector<int64_t> &output_shape,
                 std::shared_ptr<char> buffer_out,
                 std::shared_ptr<ModelExecutor> engine) {
  if (LIMIT_TIMESTAMP > 0) {
    auto now = NowMicros();
    if (now > LIMIT_TIMESTAMP) {
      return -1;
    }
  }
  std::map<std::string, deepvan::TensorWrapper> inputs;
  std::map<std::string, deepvan::TensorWrapper> outputs;
  inputs[input_name] = deepvan::TensorWrapper(input_shape, buffer_in);
  outputs[output_name] = deepvan::TensorWrapper(output_shape, buffer_out);

  // run model
  engine->Run(inputs, &outputs);
  return 0;
}

int ReleaseExecutor(std::shared_ptr<ModelExecutor> engine) {
  engine.reset();
  return 0;
}

// namespace deepvan {

#ifdef MEMPROF_SUPPORT
// For memory profiling, we will define some functions.
long getCPUMemoryUsage(std::shared_ptr<deepvan::ModelExecutor> engine) {
  if (engine == nullptr) {
    return VanState::INVALID_ARGS;
  }
  return memory_usage::getInstance().getCPUMemoryUsage();
}

long getGPUMemoryUsage(std::shared_ptr<deepvan::ModelExecutor> engine) {
  if (engine == nullptr) {
    return VanState::INVALID_ARGS;
  }
  return memory_usage::getInstance().getGPUMemoryUsage();
}
#endif // MEMPROF_SUPPORT
//}
} // namespace deepvan
