#include <numeric>
#include <stddef.h>
#include <string>
#include <vector>

#include "deepvan/compat/file_system.h"
#include "deepvan/backend/common/transpose.h"
#include "deepvan/core/types.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/export/xgen.h"
#include "deepvan/utils/logging.h"
#include "deepvan/utils/macros.h"
#include "deepvan/utils/math.h"
#include "deepvan/utils/memory.h"

using namespace deepvan;

namespace {
class XGenExecutorBase;
class XGenTensorBase;
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

struct XGenHandle {
  XGenExecutorBase *executor;
};

struct XGenTensor {
  XGenTensorBase *tensor;
};

#ifdef __cplusplus
}
#endif

namespace {
class XGenTensorBase {
public:
  virtual ~XGenTensorBase() {}

  virtual XGenType getType() const = 0;
  virtual int32_t getNumDims() const = 0;
  virtual int32_t getDim(int32_t dim_index) const = 0;
  virtual size_t getSizeInBytes() const = 0;
  virtual const void *getData() const = 0;
  virtual void *getData() = 0;
  virtual const char *getName() const = 0;

  virtual void copyFromBuffer(const void *input_data,
                              size_t input_size_in_bytes) = 0;
  virtual void copyToBuffer(void *output_data, size_t output_size_in_bytes) = 0;
};

class XGenExecutorBase {
public:
  virtual ~XGenExecutorBase() {
    for (auto tensor_ptr : input_tensors) {
      if (tensor_ptr != nullptr) {
        delete tensor_ptr->tensor;
        delete tensor_ptr;
      }
    }
    for (auto tensor_ptr : output_tensors) {
      if (tensor_ptr != nullptr) {
        delete tensor_ptr->tensor;
        delete tensor_ptr;
      }
    }
  }

  size_t getNumInputTensors() const { return input_tensors.size(); }
  const XGenTensor *getInputTensor(size_t index) const {
    return input_tensors[index];
  }
  XGenTensor *getInputTensor(size_t index) { return input_tensors[index]; }

  size_t getNumOutputTensors() { return output_tensors.size(); }
  const XGenTensor *getOutputTensor(size_t index) const {
    return output_tensors[index];
  }
  XGenTensor *getOutputTensor(size_t index) { return output_tensors[index]; }

  virtual XGenStatus Run() = 0;

protected:
  std::vector<XGenTensor *> input_tensors;
  std::vector<XGenTensor *> output_tensors;
};

#ifdef FALLBACK_SUPPORT
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

class XGenFallbackTensor : public XGenTensorBase {
public:
  XGenFallbackTensor(const TfLiteTensor *tensor)
      : tensor(const_cast<TfLiteTensor *>(tensor)) {}

  virtual ~XGenFallbackTensor() {}

public:
  XGenType getType() const override {
    return TfLiteType2XGenType(TfLiteTensorType(tensor));
  }
  int32_t getNumDims() const override { return TfLiteTensorNumDims(tensor); }
  int32_t getDim(int32_t dim_index) const override {
    return TfLiteTensorDim(tensor, dim_index);
  }
  size_t getSizeInBytes() const override {
    return TfLiteTensorByteSize(tensor);
  }
  const void *getData() const override { return TfLiteTensorData(tensor); }
  void *getData() override { return TfLiteTensorData(tensor); }
  const char *getName() const override { return TfLiteTensorName(tensor); }
  void copyFromBuffer(const void *input_data,
                      size_t input_size_in_bytes) override {
    TfLiteTensorCopyFromBuffer(tensor, input_data, input_size_in_bytes);
  }
  void copyToBuffer(void *output_data, size_t output_size_in_bytes) override {
    TfLiteTensorCopyToBuffer(tensor, output_data, output_size_in_bytes);
  }
private:
  XGenType TfLiteType2XGenType(TfLiteType type) const {
    static XGenType conv[] = {XGenNone,
                              XGenFloat32,
                              XGenNone,
                              XGenUInt8,
                              XGenNone,
                              XGenNone,
                              XGenNone,
                              XGenNone,
                              XGenNone,
                              XGenInt8,
                              XGenFloat16};
    if ((size_t)type < sizeof(conv) / sizeof(conv[0])) {
      return conv[(size_t)type];
    }
    return XGenNone;
  }

private:
  TfLiteTensor *tensor;
};

class XGenFallbackExecutor : public XGenExecutorBase {
private:
  XGenFallbackExecutor(TfLiteModel *tflite_model,
                       bool use_cpu_only = false)
      : model(tflite_model),
        options(TfLiteInterpreterOptionsCreate()),
        use_cpu_only(use_cpu_only) {
    if (!use_cpu_only) {
      TfLiteGpuDelegateOptionsV2 delegate_options(
          TfLiteGpuDelegateOptionsV2Default());
      delegate_options.is_precision_loss_allowed = 1;
      delegate_options.inference_priority1 =
          TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      delegate_options.experimental_flags =
          TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
      delegate_options.experimental_flags |=
          TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
      gpu_delegate = TfLiteGpuDelegateV2Create(&delegate_options);
      TfLiteInterpreterOptionsAddDelegate(options, gpu_delegate);
    }

    TfLiteXNNPackDelegateOptions xnnpack_options =
        TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = 4;
    xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);
    TfLiteInterpreterOptionsAddDelegate(options, xnnpack_delegate);

    TfLiteInterpreterOptionsSetNumThreads(options, 4);
    interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);

    size_t num_input_tensors =
        TfLiteInterpreterGetInputTensorCount(interpreter);
    for (size_t i = 0; i < num_input_tensors; ++i) {
      XGenTensor *t = new XGenTensor;
      TfLiteTensor *tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
      t->tensor = new XGenFallbackTensor(tensor);
      input_tensors.push_back(t);
    }
    size_t num_output_tensors =
        TfLiteInterpreterGetOutputTensorCount(interpreter);
    for (size_t i = 0; i < num_output_tensors; ++i) {
      XGenTensor *t = new XGenTensor;
      const TfLiteTensor *tensor =
          TfLiteInterpreterGetOutputTensor(interpreter, i);
      t->tensor = new XGenFallbackTensor(tensor);
      output_tensors.push_back(t);
    }
  }
public:
  XGenFallbackExecutor(const void *model_data,
                       size_t model_size,
                       bool use_cpu_only = false)
      : XGenFallbackExecutor(TfLiteModelCreate(model_data, model_size), use_cpu_only) {}
  XGenFallbackExecutor(const char* model_path,
                       bool use_cpu_only = false)
      : XGenFallbackExecutor(TfLiteModelCreateFromFile(model_path), use_cpu_only) {}

  virtual ~XGenFallbackExecutor() {
    TfLiteInterpreterDelete(interpreter);
    if (gpu_delegate)
      TfLiteGpuDelegateV2Delete(gpu_delegate);
    if (xnnpack_delegate)
      TfLiteXNNPackDelegateDelete(xnnpack_delegate);
    TfLiteModelDelete(model);
    TfLiteInterpreterOptionsDelete(options);
  }

public:
  XGenStatus Run() override {
    if (TfLiteInterpreterInvoke(interpreter) == kTfLiteOk) {
      return XGenOk;
    }
    return XGenError;
  }

private:
  TfLiteModel *model = nullptr;
  TfLiteInterpreterOptions *options = nullptr;
  TfLiteInterpreter *interpreter = nullptr;
  TfLiteDelegate *gpu_delegate = nullptr;
  TfLiteDelegate *xnnpack_delegate = nullptr;
  bool use_cpu_only = false;
};
#endif

class XGenCoCoGenExecutor;

class XGenCoCoGenTensor : public XGenTensorBase {
public:
  XGenCoCoGenTensor(XGenCoCoGenExecutor *executor,
                    const InputOutputInfo *info,
                    bool is_output);
  virtual ~XGenCoCoGenTensor() {}

  XGenType getType() const override {
    switch (info->data_type()) {
    default:
      return XGenNone;
    case DT_FLOAT: return XGenFloat32;
    case DT_HALF: return XGenFloat16;
    case DT_INT32: return XGenInt32;
    case DT_INT8: return XGenInt8;
    case DT_UINT8: return XGenUInt8;
    }
  }
  int32_t getNumDims() const override { return info->dims().size(); }
  int32_t getDim(int32_t dim_index) const override {
    if (info->data_format() == DataFormat::NHWC && info->dims().size() == 4) {
      // info->dims() returns dimension info in internal NHWC format but
      // users expect NCHW for input and output.
      const unsigned NHWC2NCHW[] = {0, 3, 1, 2};
      return info->dims()[NHWC2NCHW[dim_index]];
    } else {
      return info->dims()[dim_index];
    }
  }
  size_t getSizeInBytes() const override {
    return getNumElements() * sizeOf(getType());
  }
  const void *getData() const override { return nullptr; }
  void *getData() override { return nullptr; }
  const char *getName() const override { return info->name().c_str(); }

  void copyFromBuffer(const void *input_data,
                      size_t input_size_in_bytes) override;
  void copyToBuffer(void *output_data, size_t output_size_in_bytes) override {
    auto tensor = getOutputTensor();
    size_t output_size = getSizeInBytes();
    if (info->data_format() == DataFormat::NHWC && info->dims().size() == 4) {
      // FIXME: TransposeOutput does not work because output tensor does
      // not have data format.
      std::vector<int> dst_dims = {0, 3, 1, 2};
      std::vector<index_t> output_shape(info->dims().begin(),
                                        info->dims().end());
      CONDITIONS(output_size <= output_size_in_bytes)
          << "Output size exceeds buffer size: shape"
          << MakeString<int64_t>(output_shape) << " vs buffer size "
          << output_size_in_bytes;
      Transpose<float>(tensor->data().get(),
                       output_shape,
                       dst_dims,
                       reinterpret_cast<float *>(output_data));
    } else {
      // if (output_size % 16 == 0 && (output_size / 16) % 4 == 0) {
      //   deepvan::neon_memcpy(reinterpret_cast<float *>(output_data),
      //                        tensor->data().get(),
      //                        output_size);
      // } else {
        deepvan::fast_memcpy(output_data, tensor->data().get(), output_size);
      // }
    }
  }

  const TensorWrapper *getInputTensor() const;
  TensorWrapper *getInputTensor();
  const TensorWrapper *getOutputTensor() const;
  TensorWrapper *getOutputTensor();

private:
  size_t sizeOf(XGenType type) const {
    size_t size[] = {0, 4, 2, 4, 1, 1};
    return size[(size_t)type];
  }
  size_t getNumElements() const {
    return std::accumulate(
        info->dims().begin(), info->dims().end(), 1, std::multiplies<size_t>());
  }

private:
  XGenCoCoGenExecutor *executor;
  InputOutputInfo *info;
};

class XGenCoCoGenExecutor : public XGenExecutorBase {
public:
  XGenCoCoGenExecutor(const void *model_data,
                      size_t model_size_in_bytes,
                      const void *extra_data,
                      size_t data_size_in_bytes,
                      XGenPowerPolicy policy = XGenPowerDefault)
      : net_def(std::make_shared<NetProto>()), power_policy(policy) {
    UNUSED_VARIABLE(data_size_in_bytes);

    net_def->ParseFromArray(model_data, model_size_in_bytes);

    std::vector<std::string> input_nodes, output_nodes;
    for (auto &input_info : net_def->input_info()) {
      XGenTensor *t = new XGenTensor;
      XGenCoCoGenTensor *input =
          new XGenCoCoGenTensor(this, &input_info, false);
      std::string name(input->getName());
      t->tensor = input;
      input_tensors.push_back(t);
      input_nodes.push_back(name);
    }
    for (auto &output_info : net_def->output_info()) {
      XGenTensor *t = new XGenTensor;
      XGenCoCoGenTensor *output =
          new XGenCoCoGenTensor(this, &output_info, true);
      std::string name(output->getName());
      t->tensor = output;
      output_tensors.push_back(t);
      output_nodes.push_back(name);
    }

    ModelExecutionConfig config(DeviceType::GPU);

    switch (power_policy) {
    default:
      break;
    case XGenPowerNone:
      break;
    case XGenPowerDefault:
      config.SetCPUThreadPolicy(4, CPUAffinityPolicy::AFFINITY_BIG_ONLY);
      break;
    case XGenPowerPerformance:
      config.SetCPUThreadPolicy(2, CPUAffinityPolicy::AFFINITY_PERFORMANCE);
      break;
    case XGenPowerSave:
      config.SetCPUThreadPolicy(2, CPUAffinityPolicy::AFFINITY_POWER_SAVE);
      break;
    }

    // const char *storage_path_ptr = getenv("DEEPVAN_INTERNAL_STORAGE_PATH");
    // const std::string storage_path = std::string(
    //     storage_path_ptr == nullptr ? "/data/local/tmp/deepvan_run/interior"
    //                                 : storage_path_ptr);
    std::vector<std::string> opencl_binary_paths = {""};

    constexpr int hint = 3;
    config.SetGPUContext(GPUContextBuilder()
                             .SetStoragePath("")
                             // The directory above is not accessible to the app like cocoplayer
                             // .SetStoragePath(storage_path)
                             .SetOpenCLBinaryPaths(opencl_binary_paths)
                             .SetOpenCLParameterPath("")
                             .Finalize());
    config.SetGPUHints(static_cast<GPUPerfHint>(hint),
                       static_cast<GPUPriorityHint>(hint));

    engine.reset(new deepvan::ModelExecutor(config));
    VanState status =
        engine->Init(net_def.get(),
                     input_nodes,
                     output_nodes,
                     reinterpret_cast<const unsigned char *>(extra_data));
  }
  virtual ~XGenCoCoGenExecutor() {}

  XGenStatus Run() override {
    if (engine->Run(inputs, &outputs) == VanState::SUCCEED) {
      return XGenOk;
    }
    return XGenError;
  }

  void addInput(const std::string &name, const XGenCoCoGenTensor &tensor) {
    int32_t num_dims = tensor.getNumDims();
    std::vector<int64_t> shape;
    size_t num_floats = 1;
    for (int32_t i = 0; i < num_dims; ++i) {
      int32_t dim = tensor.getDim(i);
      shape.push_back(dim);
      num_floats *= dim;
    }
    std::shared_ptr<float> buffer = std::shared_ptr<float>(
        new float[num_floats], std::default_delete<float>());
    inputs[name] = TensorWrapper(shape, buffer, DataFormat::NCHW);
  }
  void addOutput(const std::string &name, const XGenCoCoGenTensor &tensor) {
    int32_t num_dims = tensor.getNumDims();
    std::vector<int64_t> shape;
    size_t num_floats = 1;
    for (int32_t i = 0; i < num_dims; ++i) {
      int32_t dim = tensor.getDim(i);
      shape.push_back(dim);
      num_floats *= dim;
    }
    std::shared_ptr<float> buffer = std::shared_ptr<float>(
        new float[num_floats], std::default_delete<float>());
    outputs[name] = TensorWrapper(shape, buffer, DataFormat::NCHW);
  }

  const TensorWrapper *getInputTensor(const XGenCoCoGenTensor &tensor) const {
    std::string name(tensor.getName());
    auto it = inputs.find(name);
    if (it == inputs.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }
  TensorWrapper *getInputTensor(const XGenCoCoGenTensor &tensor) {
    std::string name(tensor.getName());
    auto it = inputs.find(name);
    if (it == inputs.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

  const TensorWrapper *getOutputTensor(const XGenCoCoGenTensor &tensor) const {
    std::string name(tensor.getName());
    auto it = outputs.find(name);
    if (it == outputs.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }
  TensorWrapper *getOutputTensor(const XGenCoCoGenTensor &tensor) {
    std::string name(tensor.getName());
    auto it = outputs.find(name);
    if (it == outputs.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

private:
  std::shared_ptr<deepvan::NetProto> net_def;
  std::shared_ptr<deepvan::ModelExecutor> engine;

  std::map<std::string, deepvan::TensorWrapper> inputs;
  std::map<std::string, deepvan::TensorWrapper> outputs;

  XGenPowerPolicy power_policy;
};

XGenCoCoGenTensor::XGenCoCoGenTensor(XGenCoCoGenExecutor *executor,
                                     const InputOutputInfo *info,
                                     bool is_output)
    : executor(executor), info(const_cast<InputOutputInfo *>(info)) {
  if (is_output) {
    executor->addOutput(std::string(getName()), *this);
  }
}

void XGenCoCoGenTensor::copyFromBuffer(const void *input_data,
                                       size_t input_size_in_bytes) {
  // Reallocate input?
  executor->addInput(std::string(getName()), *this);
  auto tensor = getInputTensor();
  memcpy(tensor->data().get(), input_data, input_size_in_bytes);
}

const TensorWrapper *XGenCoCoGenTensor::getInputTensor() const {
  return executor->getInputTensor(*this);
}
TensorWrapper *XGenCoCoGenTensor::getInputTensor() {
  return executor->getInputTensor(*this);
}
const TensorWrapper *XGenCoCoGenTensor::getOutputTensor() const {
  return executor->getOutputTensor(*this);
}
TensorWrapper *XGenCoCoGenTensor::getOutputTensor() {
  return executor->getOutputTensor(*this);
}

// To generate a new timestamp from command line:
//   $ date -d "2022-10-31" +%s
//   1667199600
const long long expiration_unix_timestamp = 1682899199;  // Sun Apr 30 2023 23:59:59 GMT+0000

bool has_license_expired() {
  auto current_time = std::chrono::system_clock::now();
  auto current_unix_timestamp_sec =
      std::chrono::duration_cast<std::chrono::seconds>(
          current_time.time_since_epoch())
          .count();
  std::cout << "Current expiration date is (Sun Apr 30 2023 23:59:59 GMT+0000)" << std::endl;

  return current_unix_timestamp_sec >= expiration_unix_timestamp;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

XGEN_EXPORT XGenHandle *XGenInitWithFiles(const char *model_file, 
                                          const char *data_file, 
                                          XGenPowerPolicy policy) {
  std::unique_ptr<deepvan::compat::ReadOnlyMemoryRegion> model_graph_data =
      make_unique<deepvan::compat::ReadOnlyBufferMemoryRegion>();
  auto fs = GetFileSystem();
  VanState status = fs->NewReadOnlyMemoryRegionFromFile(model_file, &model_graph_data);
  if (status != VanState::SUCCEED) {
    LOG(FATAL) << "Failed to read file: " << model_file;
  }
  

  std::unique_ptr<deepvan::compat::ReadOnlyMemoryRegion> model_weights_data =
      make_unique<deepvan::compat::ReadOnlyBufferMemoryRegion>();
  
  fs = GetFileSystem();
  status = fs->NewReadOnlyMemoryRegionFromFile(data_file, &model_weights_data);
  if (status != VanState::SUCCEED) {
    LOG(FATAL) << "Failed to read file: " << data_file;
  }

  return XGenInitWithPower(reinterpret_cast<const unsigned char *>(model_graph_data->data()),
                          model_graph_data->length(),
                          reinterpret_cast<const unsigned char *>(model_weights_data->data()),
                          model_weights_data->length(),
                          policy);
}

XGEN_EXPORT XGenHandle *XGenInit(const void *model_data, size_t model_size) {
  if (has_license_expired()) {
    LOG(ERROR) << "The license for XGen inference engine has expired.";
    return nullptr;
  }

  XGenHandle *handle = new XGenHandle;
#ifdef FALLBACK_SUPPORT
  handle->executor = new XGenFallbackExecutor(model_data, model_size);
#else
  handle->executor = nullptr;
#endif
  return handle;
}

XGEN_EXPORT XGenHandle *XGenInitWithCPUOnly(const void *model_data,
                                            size_t model_size) {
  if (has_license_expired()) {
    LOG(ERROR) << "The license for XGen inference engine has expired.";
    return nullptr;
  }

  XGenHandle *handle = new XGenHandle;
#ifdef FALLBACK_SUPPORT
  handle->executor = new XGenFallbackExecutor(model_data, model_size, true);
#else
  handle->executor = nullptr;
#endif
  return handle;
}

XGEN_EXPORT XGenHandle *XGenInitWithData(const void *model_data,
                                         size_t model_size_in_bytes,
                                         const void *extra_data,
                                         size_t data_size_in_bytes) {
  if (has_license_expired()) {
    LOG(ERROR) << "The license for XGen inference engine has expired.";
    return nullptr;
  }

  XGenHandle *handle = new XGenHandle;
  handle->executor = new XGenCoCoGenExecutor(
      model_data, model_size_in_bytes, extra_data, data_size_in_bytes);
  return handle;
}

XGEN_EXPORT XGenHandle *XGenInitWithPower(const void *model_data,
                                          size_t model_size_in_bytes,
                                          const void *extra_data,
                                          size_t data_size_in_bytes,
                                          XGenPowerPolicy policy) {
  if (has_license_expired()) {
    LOG(ERROR) << "The license for XGen inference engine has expired.";
    return nullptr;
  }

  XGenHandle *handle = new XGenHandle;
  handle->executor = new XGenCoCoGenExecutor(
      model_data, model_size_in_bytes, extra_data, data_size_in_bytes, policy);
  return handle;
}

XGEN_EXPORT XGenHandle *XGenInitWithFallbackFiles(const char *model_path) {
  if (has_license_expired()) {
    std::cout << "The license for XGen inference engine has expired.";
    return nullptr;
  }

  XGenHandle *handle = new XGenHandle;
#ifdef FALLBACK_SUPPORT
  handle->executor = new XGenFallbackExecutor(model_path);
#else
  UNUSED_VARIABLE(model_path);
  handle->executor = nullptr;
#endif
  return handle;
}

XGEN_EXPORT XGenStatus XGenRun(XGenHandle *handle) {
  if (handle == nullptr || handle->executor == nullptr) {
    return XGenError;
  }

  if (has_license_expired()) {
    LOG(ERROR) << "The license for XGen inference engine has expired.";
    return XGenLicenseExpired;
  }

  return handle->executor->Run();
}

XGEN_EXPORT void XGenShutdown(XGenHandle *handle) {
  if (handle != nullptr) {
    if (handle->executor != nullptr) {
      delete handle->executor;
    }
    delete handle;
  }
}

XGEN_EXPORT size_t XGenGetNumInputTensors(const XGenHandle *handle) {
  if (handle == nullptr || handle->executor == nullptr) {
    return 0;
  }
  return handle->executor->getNumInputTensors();
}
XGEN_EXPORT size_t XGenGetNumOutputTensors(const XGenHandle *handle) {
  if (handle == nullptr || handle->executor == nullptr) {
    return 0;
  }
  return handle->executor->getNumOutputTensors();
}
XGEN_EXPORT XGenTensor *XGenGetInputTensor(XGenHandle *handle,
                                           size_t tensor_index) {
  if (handle == nullptr || handle->executor == nullptr) {
    return nullptr;
  }
  return handle->executor->getInputTensor(tensor_index);
}
XGEN_EXPORT XGenTensor *XGenGetOutputTensor(XGenHandle *handle,
                                            size_t tensor_index) {
  if (handle == nullptr || handle->executor == nullptr) {
    return nullptr;
  }
  return handle->executor->getOutputTensor(tensor_index);
}
XGEN_EXPORT void XGenCopyBufferToTensor(XGenTensor *input_tensor,
                                        const void *input_data,
                                        size_t input_size_in_bytes) {
  if (input_tensor != nullptr && input_tensor->tensor != nullptr) {
    input_tensor->tensor->copyFromBuffer(input_data, input_size_in_bytes);
  }
}
XGEN_EXPORT void XGenCopyTensorToBuffer(const XGenTensor *output_tensor,
                                        void *output_data,
                                        size_t output_size_in_bytes) {
  if (output_tensor != nullptr && output_tensor->tensor != nullptr) {
    output_tensor->tensor->copyToBuffer(output_data, output_size_in_bytes);
  }
}

XGEN_EXPORT XGenType XGenGetTensorType(const XGenTensor *tensor) {
  if (tensor == nullptr || tensor->tensor == nullptr) {
    return XGenNone;
  }
  return tensor->tensor->getType();
}
XGEN_EXPORT int32_t XGenGetTensorNumDims(const XGenTensor *tensor) {
  if (tensor == nullptr || tensor->tensor == nullptr) {
    return 0;
  }
  return tensor->tensor->getNumDims();
}
XGEN_EXPORT int32_t XGenGetTensorDim(const XGenTensor *tensor,
                                     int32_t dim_index) {
  if (tensor == nullptr || tensor->tensor == nullptr) {
    return 0;
  }
  return tensor->tensor->getDim(dim_index);
}
XGEN_EXPORT size_t XGenGetTensorSizeInBytes(const XGenTensor *tensor) {
  if (tensor == nullptr || tensor->tensor == nullptr) {
    return 0;
  }
  return tensor->tensor->getSizeInBytes();
}
XGEN_EXPORT void *XGenGetTensorData(const XGenTensor *tensor) {
  if (tensor == nullptr || tensor->tensor == nullptr) {
    return nullptr;
  }
  return tensor->tensor->getData();
}
XGEN_EXPORT const char *XGenGetTensorName(const XGenTensor *tensor) {
  if (tensor == nullptr || tensor->tensor == nullptr) {
    return nullptr;
  }
  return tensor->tensor->getName();
}
#ifdef __cplusplus
}
#endif
