#ifndef DEEPVAN_CORE_OPERATOR_H_
#define DEEPVAN_CORE_OPERATOR_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "deepvan/core/arg_helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include "deepvan/core/network_controller.h"
#include "deepvan/proto/deepvan.pb.h"
#ifdef OPENCL_SUPPORT
#include "deepvan/core/runtime/opencl/opencl_shape_util.h"
#endif // OPENCL_SUPPORT

namespace deepvan {
// memory_optimizer, device
class OpConstructContext {
  typedef std::unordered_map<std::string, std::vector<index_t>> TensorShapeMap;

public:
  explicit OpConstructContext(NetworkController *ws);
  OpConstructContext(NetworkController *ws, TensorShapeMap *info);
  ~OpConstructContext() = default;

  void set_operator_def(std::shared_ptr<OperatorProto> operator_def);

  inline std::shared_ptr<OperatorProto> operator_def() const {
    return operator_def_;
  }

  inline NetworkController *workspace() const { return ws_; }

  inline void set_device(Device *device) { device_ = device; }

  inline Device *device() const { return device_; }

  inline TensorShapeMap *tensor_shape_info() const {
    return tensor_shape_info_;
  }

  void set_output_mem_type(MemoryType type);

  inline MemoryType output_mem_type() const { return output_mem_type_; }

  void SetInputInfo(size_t idx, MemoryType mem_type, DataType dt);

  MemoryType GetInputMemType(size_t idx) const;

  DataType GetInputDataType(size_t idx) const;

#ifdef OPENCL_SUPPORT
  void SetInputOpenCLBufferType(size_t idx, OpenCLBufferType buffer_type);
  OpenCLBufferType GetInputOpenCLBufferType(size_t idx) const;
#endif // OPENCL_SUPPORT

private:
  std::shared_ptr<OperatorProto> operator_def_;
  NetworkController *ws_;
  Device *device_;
  TensorShapeMap *tensor_shape_info_;
  // used for memory transform
  std::vector<MemoryType> input_mem_types_;
  std::vector<DataType> input_data_types_;
  MemoryType output_mem_type_; // there is only one output memory type now.
#ifdef OPENCL_SUPPORT
  std::vector<OpenCLBufferType> input_opencl_buffer_types_;
#endif // OPENCL_SUPPORT
};

// memory_optimizer, device
class OpInitContext {
public:
  explicit OpInitContext(NetworkController *ws, Device *device = nullptr);
  ~OpInitContext() = default;

  inline NetworkController *workspace() const { return ws_; }

  inline void set_device(Device *device) { device_ = device; }

  inline Device *device() const { return device_; }

private:
  NetworkController *ws_;
  Device *device_;
};

// Conventions
// * If there exist format, NHWC is the default format
// * The input/output format of CPU ops with float data type is NCHW
// * The input/output format of GPU ops and CPU Quantization ops is NHWC
// * Inputs' data type is same as the operation data type by default.
// * The outputs' data type is same as the operation data type by default.
class Operation {
public:
  explicit Operation(OpConstructContext *context);
  virtual ~Operation() = default;

  template <typename T>
  inline T GetOptionalArg(const std::string &name,
                          const T &default_value) const {
    CONDITIONS(operator_def_, "operator_def was null!");
    return ProtoArgHelper::GetOptionalArg<OperatorProto, T>(*operator_def_, name,
                                                          default_value);
  }
  template <typename T>
  inline std::vector<T>
  GetRepeatedArgs(const std::string &name,
                  const std::vector<T> &default_value = {}) const {
    CONDITIONS(operator_def_, "operator_def was null!");
    return ProtoArgHelper::GetRepeatedArgs<OperatorProto, T>(*operator_def_, name,
                                                           default_value);
  }

  inline DeviceType device_type() const {
    return static_cast<DeviceType>(operator_def_->device_type());
  }

  inline const Tensor *Input(unsigned int idx) {
    CONDITIONS(idx < inputs_.size());
    return inputs_[idx];
  }

  inline Tensor *Output(int idx) { return outputs_[idx]; }

  inline int InputSize() { return inputs_.size(); }
  inline int OutputSize() { return outputs_.size(); }
  inline const std::vector<const Tensor *> &Inputs() const { return inputs_; }
  inline const std::vector<Tensor *> &Outputs() { return outputs_; }

  // Run Op asynchronously (depends on device), return a future if not nullptr.
  virtual VanState Init(OpInitContext *);
  virtual VanState Run(OpContext *) = 0;

  inline const OperatorProto &debug_def() const {
    CONDITIONS(has_debug_def(), "operator_def was null!");
    return *operator_def_;
  }

  inline void set_debug_def(const std::shared_ptr<OperatorProto> &operator_def) {
    operator_def_ = operator_def;
  }

  inline bool has_debug_def() const { return operator_def_ != nullptr; }

  inline std::shared_ptr<OperatorProto> operator_def() { return operator_def_; }

  inline deepvan::PruningType get_pruning_type() {
    return operator_def_->pruning_type();
  }

  inline ModelType get_model_type() {
    return model_type_;
  }
protected:
  inline std::string operator_name() { return operator_def_->name(); }

  std::shared_ptr<OperatorProto> operator_def_;
  std::vector<const Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
  PruningType pruning_type_;
  ModelType model_type_;

  DISABLE_COPY_AND_ASSIGN(Operation);
};

// DEEPVAN_OP_INPUT_TAGS and DEEPVAN_OP_OUTPUT_TAGS are optional features to
// name the indices of the operator's inputs and outputs, in order to avoid
// confusion. For example, for a fully convolution layer that has input, weight
// and bias, you can define its input tags as:
//     DEEPVAN_OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
// And in the code, instead of doing
//     auto& weight = Input(1);
// you can now do
//     auto& weight = Input(WEIGHT);
// to make it more clear.
#define DEEPVAN_OP_INPUT_TAGS(first_input, ...)                                \
  enum _InputTags { first_input = 0, __VA_ARGS__ }
#define DEEPVAN_OP_OUTPUT_TAGS(first_input, ...)                               \
  enum _OutputTags { first_input = 0, __VA_ARGS__ }

struct OpRegistrationInfo {
public:
  typedef std::function<std::unique_ptr<Operation>(OpConstructContext *)>
      OpCreator;
  typedef std::function<std::set<DeviceType>(OpConstructContext *)>
      DevicePlacer;

  OpRegistrationInfo();

  void AddDevice(DeviceType);

  void Register(const std::string &key, OpCreator creator);

  std::set<DeviceType> devices;
  std::unordered_map<std::string, OpCreator> creators;
  DevicePlacer device_placer;
};

class OpConditionBuilder {
public:
  explicit OpConditionBuilder(const std::string &type);

  const std::string type() const;

  OpConditionBuilder &
  SetDevicePlacerFunc(OpRegistrationInfo::DevicePlacer placer);

  void Finalize(OpRegistrationInfo *info) const;

private:
  std::string type_;
  OpRegistrationInfo::DevicePlacer placer_;
};

class OpRegistryBase {
public:
  OpRegistryBase() = default;
  virtual ~OpRegistryBase() = default;
  VanState Register(const std::string &op_type, const DeviceType device_type,
                    const DataType dt, OpRegistrationInfo::OpCreator creator);

  VanState Register(const OpConditionBuilder &builder);

  const std::set<DeviceType>
  AvailableDevices(const std::string &op_type,
                   OpConstructContext *context) const;

  std::unique_ptr<Operation> CreateOperation(OpConstructContext *context,
                                             DeviceType device_type) const;

  template <class DerivedType>
  static std::unique_ptr<Operation>
  DefaultCreator(OpConstructContext *context) {
    return std::unique_ptr<Operation>(new DerivedType(context));
  }

private:
  std::unordered_map<std::string, std::unique_ptr<OpRegistrationInfo>>
      registry_;
  DISABLE_COPY_AND_ASSIGN(OpRegistryBase);
};

#define VAN_REGISTER_OP(op_registry, op_type, class_name, device, dt)          \
  op_registry->Register(                                                       \
      op_type, device, DataTypeToEnum<dt>::value,                              \
      OpRegistryBase::DefaultCreator<class_name<device, dt>>)

#define VAN_REGISTER_OP_CONDITION(op_registry, builder)                        \
  op_registry->Register(builder)

#define VAN_REGISTER_OP_BY_EXECUTING_ON(op_registry, name, default_devices)    \
  op_registry->Register(OpConditionBuilder(name).SetDevicePlacerFunc(          \
      [](OpConstructContext *context) -> std::set<DeviceType> {                \
        std::set<DeviceType> result;                                           \
        auto op_def = context->operator_def();                                 \
        auto executing_on = ProtoArgHelper::GetRepeatedArgs<OperatorProto, int>( \
            *op_def, "executing_on", default_devices);                         \
        for (auto exe : executing_on) {                                        \
          result.insert(static_cast<DeviceType>(exe));                         \
        }                                                                      \
        return result;                                                         \
      }))

#define VAN_REGISTER_OP_BY_EXECUTING_ON1(op_registry, name, default_devices)   \
  op_registry->Register(OpConditionBuilder(name).SetDevicePlacerFunc(          \
      [](OpConstructContext *context) -> std::set<DeviceType> {                \
        std::set<DeviceType> result;                                           \
        auto op_def = context->operator_def();                                 \
        auto executing_on = ProtoArgHelper::GetRepeatedArgs<OperatorProto, int>( \
            *op_def, "executing_on", default_devices);                         \
        for (auto exe : executing_on) {                                        \
          result.insert(static_cast<DeviceType>(exe));                         \
        }                                                                      \
        return result;                                                         \
      }))

} // namespace deepvan

#endif // DEEPVAN_CORE_OPERATOR_H_
