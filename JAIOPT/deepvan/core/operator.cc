#include <sstream>
#include <map>
#include <memory>
#include <vector>

#include "deepvan/core/operator.h"

namespace deepvan {
OpConstructContext::OpConstructContext(NetworkController *ws)
    : operator_def_(nullptr),
      ws_(ws),
      device_(nullptr),
      tensor_shape_info_(nullptr) {}

OpConstructContext::OpConstructContext(
    deepvan::NetworkController *ws,
    deepvan::OpConstructContext::TensorShapeMap *info)
    : operator_def_(nullptr),
      ws_(ws),
      device_(nullptr),
      tensor_shape_info_(info) {}

void OpConstructContext::set_operator_def(
    std::shared_ptr<deepvan::OperatorProto> operator_def) {
  operator_def_ = operator_def;
  input_data_types_.clear();
}

void OpConstructContext::set_output_mem_type(deepvan::MemoryType type) {
  CONDITIONS(operator_def_ != nullptr);
  output_mem_type_ = type;
  input_mem_types_.clear();
}

void OpConstructContext::SetInputInfo(size_t idx,
                                      deepvan::MemoryType mem_type,
                                      deepvan::DataType dt) {
  if (input_mem_types_.empty()) {
    // the default inputs' memory types are same as output memory type.
    input_mem_types_.resize(operator_def_->input_size(), output_mem_type_);
  }
  if (input_data_types_.empty()) {
    // the default inputs' data types are same as operation's data type.
    DataType op_dt = static_cast<DataType>(
        ProtoArgHelper::GetOptionalArg<OperatorProto, int>(
            *operator_def_, "T", static_cast<int>(DataType::DT_FLOAT)));
    input_data_types_.resize(operator_def_->input_size(), op_dt);
  }
  CONDITIONS(idx < input_mem_types_.size() && idx < input_data_types_.size());
  input_mem_types_[idx] = mem_type;
  input_data_types_[idx] = dt;
}

MemoryType OpConstructContext::GetInputMemType(size_t idx) const {
  if (input_mem_types_.empty()) {
    return output_mem_type_;
  }
  CONDITIONS(idx < input_mem_types_.size(),
             idx, " < ", input_mem_types_.size());
  return input_mem_types_[idx];
}

DataType OpConstructContext::GetInputDataType(size_t idx) const {
  if (input_data_types_.empty()) {
    // the default inputs' data types are same as operation's data type.
    return static_cast<DataType>(
        ProtoArgHelper::GetOptionalArg<OperatorProto, int>(
            *operator_def_, "T", static_cast<int>(DataType::DT_FLOAT)));
  }
  CONDITIONS(idx < input_data_types_.size());
  return input_data_types_[idx];
}

#ifdef OPENCL_SUPPORT
void OpConstructContext::SetInputOpenCLBufferType(
    size_t idx, OpenCLBufferType buffer_type) {
  if (input_opencl_buffer_types_.empty()) {
    // the default inputs' memory types are same as output memory type.
    input_opencl_buffer_types_.resize(operator_def_->input_size(),
                               OpenCLBufferType::IN_OUT_CHANNEL);
  }
  CONDITIONS(idx < input_opencl_buffer_types_.size());
  input_opencl_buffer_types_[idx] = buffer_type;
}
OpenCLBufferType OpConstructContext::GetInputOpenCLBufferType(
    size_t idx) const {
  if (input_opencl_buffer_types_.empty()) {
    return OpenCLBufferType::IN_OUT_CHANNEL;
  }
  CONDITIONS(idx < input_opencl_buffer_types_.size());
  return input_opencl_buffer_types_[idx];
}
#endif  // OPENCL_SUPPORT

OpInitContext::OpInitContext(NetworkController *ws, Device *device)
    : ws_(ws), device_(device) {}

Operation::Operation(OpConstructContext *context)
    : operator_def_(context->operator_def()), 
      pruning_type_(static_cast<PruningType>(operator_def_->pruning_type())),
      model_type_(static_cast<ModelType>(operator_def_->model_type())) {}

VanState Operation::Init(OpInitContext *context) {
  NetworkController *ws = context->workspace();
  for (const std::string &input_str : operator_def_->input()) {
    const Tensor *tensor = ws->GetTensor(input_str);
    CONDITIONS(tensor != nullptr, "op ", operator_def_->type(),
               ": Encountered a non-existing input tensor: ", input_str);
    inputs_.push_back(tensor);
  }
  for (int i = 0; i < operator_def_->output_size(); ++i) {
    const std::string output_str = operator_def_->output(i);
    if (ws->HasTensor(output_str)) {
      outputs_.push_back(ws->GetTensor(output_str));
    } else {
      CONDITIONS(
          operator_def_->output_type_size() == 0 ||
              operator_def_->output_size() == operator_def_->output_type_size(),
          "operator output size != operator output type size",
          operator_def_->output_size(),
          operator_def_->output_type_size());
      DataType output_type;
      if (i < operator_def_->output_type_size()) {
        output_type = operator_def_->output_type(i);
      } else {
        output_type = static_cast<DataType>(
            ProtoArgHelper::GetOptionalArg<OperatorProto, int>(
            *operator_def_, "T", static_cast<int>(DT_FLOAT)));
      }
      outputs_.push_back(CONDITIONS_NOTNULL(ws->CreateTensor(
          output_str, context->device()->allocator(), output_type)));
    }
    if (i < operator_def_->output_shape_size()) {
      std::vector<index_t>
          shape_configured(operator_def_->output_shape(i).dims_size());
      for (size_t dim = 0; dim < shape_configured.size(); ++dim) {
        shape_configured[dim] = operator_def_->output_shape(i).dims(dim);
      }
      ws->GetTensor(output_str)->SetShapeConfigured(shape_configured);
    }
  }
  return VanState::SUCCEED;
}

// op registry
namespace {
class OpKeyBuilder {
 public:
  explicit OpKeyBuilder(const std::string &op_name);

  OpKeyBuilder &Device(DeviceType device);

  OpKeyBuilder &TypeConstraint(const char *attr_name,
                               DataType allowed);

  const std::string Build();

 private:
  std::string op_name_;
  DeviceType device_type_;
  std::map<std::string, DataType> type_constraint_;
};

OpKeyBuilder::OpKeyBuilder(const std::string &op_name) : op_name_(op_name) {}

OpKeyBuilder &OpKeyBuilder::Device(DeviceType device) {
  device_type_ = device;
  return *this;
}

OpKeyBuilder &OpKeyBuilder::TypeConstraint(const char *attr_name,
                                           DataType allowed) {
  type_constraint_[attr_name] = allowed;
  return *this;
}

const std::string OpKeyBuilder::Build() {
  static const std::vector<std::string> type_order = {"T"};
  std::stringstream ss;
  ss << op_name_;
  ss << device_type_;
  for (auto type : type_order) {
    ss << type << "_" << DataTypeToString(type_constraint_[type]);
  }

  return ss.str();
}
}  // namespace

OpRegistrationInfo::OpRegistrationInfo() {
  device_placer = [this](OpConstructContext *context) -> std::set<DeviceType> {
    auto op = context->operator_def();
    // TODO @vgod
    // The GPU ops only support 4D In/Out tensor by default
    if (this->devices.count(DeviceType::CPU) == 1 &&
        op->output_shape_size() == op->output_size() &&
        op->output_shape(0).dims_size() != 4) {
      return { DeviceType::CPU };
    }
    return this->devices;
  };
}

void OpRegistrationInfo::AddDevice(deepvan::DeviceType device) {
  devices.insert(device);
}

void OpRegistrationInfo::Register(const std::string &key, OpCreator creator) {
  VLOG(3) << "Registering: " << key;
  CONDITIONS(creators.count(key) == 0, "Key already registered: ", key);
  creators[key] = creator;
}

VanState OpRegistryBase::Register(
    const std::string &op_type,
    const deepvan::DeviceType device_type,
    const deepvan::DataType dt,
    deepvan::OpRegistrationInfo::OpCreator creator) {
  if (registry_.count(op_type) == 0) {
    registry_[op_type] = std::unique_ptr<OpRegistrationInfo>(
        new OpRegistrationInfo);
  }
  registry_[op_type]->AddDevice(device_type);

  std::string op_key = OpKeyBuilder(op_type)
      .Device(device_type)
      .TypeConstraint("T", dt)
      .Build();
  registry_.at(op_type)->Register(op_key, creator);
  return VanState::SUCCEED;
}

VanState OpRegistryBase::Register(
    const OpConditionBuilder &builder) {
  std::string op_type = builder.type();
  if (registry_.count(op_type) == 0) {
    registry_[op_type] = std::unique_ptr<OpRegistrationInfo>(
        new OpRegistrationInfo);
  }
  builder.Finalize(registry_[op_type].get());
  return VanState::SUCCEED;
}

const std::set<DeviceType> OpRegistryBase::AvailableDevices(
    const std::string &op_type, OpConstructContext *context) const {
  CONDITIONS_OP(registry_.count(op_type) != 0, op_type);

  return registry_.at(op_type)->device_placer(context);
}

std::unique_ptr<Operation> OpRegistryBase::CreateOperation(
    OpConstructContext *context,
    DeviceType device_type) const {
  auto operator_def = context->operator_def();
  DataType dtype = static_cast<DataType>(
      ProtoArgHelper::GetOptionalArg<OperatorProto, int>(
          *operator_def, "T", static_cast<int>(DT_FLOAT)));
  if (device_type == DeviceType::CPU && dtype == DT_HALF) {
    int arg_size = operator_def->arg_size();
    for (int i = 0; i < arg_size; ++i) {
      if (operator_def->arg(i).name() == "T") {
        operator_def->mutable_arg(i)->set_i(DT_FLOAT);
      }
    }
    dtype = DT_FLOAT;
  }
  VLOG(1) << "Creating operator " << operator_def->name() << "("
          << operator_def->type() << "<" << dtype << ">" << ") on "
          << device_type;
  const std::string op_type = context->operator_def()->type();
  CONDITIONS(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");

  std::string key = OpKeyBuilder(op_type)
      .Device(device_type)
      .TypeConstraint("T", dtype)
      .Build();
  if (registry_.at(op_type)->creators.count(key) == 0) {
    LOG(FATAL) << "Key not registered: " << key;
  }
  return registry_.at(op_type)->creators.at(key)(context);
}

OpConditionBuilder::OpConditionBuilder(const std::string &type)
  : type_(type) {}

const std::string OpConditionBuilder::type() const {
  return type_;
}

OpConditionBuilder &OpConditionBuilder::SetDevicePlacerFunc(
    OpRegistrationInfo::DevicePlacer placer) {
  placer_ = placer;
  return *this;
}

void OpConditionBuilder::Finalize(OpRegistrationInfo *info) const {
  if (info != nullptr && placer_) {
    info->device_placer = placer_;
  }
}
}  // namespace deepvan
