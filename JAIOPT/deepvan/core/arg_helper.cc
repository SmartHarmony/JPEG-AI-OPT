#include <string>
#include <vector>

#include "deepvan/core/arg_helper.h"
#include "deepvan/utils/logging.h"

namespace deepvan {
ProtoArgHelper::ProtoArgHelper(const OperatorProto &def) {
  for (auto &arg : def.arg()) {
    if (arg_map_.count(arg.name())) {
      LOG(WARNING) << "Duplicated argument " << arg.name()
                   << " found in operator " << def.name();
    }
    arg_map_[arg.name()] = arg;
  }
}

ProtoArgHelper::ProtoArgHelper(const NetProto &NetProto) {
  for (auto &arg : NetProto.arg()) {
    CONDITIONS(arg_map_.count(arg.name()) == 0,
               "Duplicated argument found in net def.");
    arg_map_[arg.name()] = arg;
  }
}

namespace {
template <typename InputType, typename TargetType>
inline bool IsCastLossless(const InputType &value) {
  return static_cast<InputType>(static_cast<TargetType>(value)) == value;
}
} // namespace

#define GET_OPTIONAL_ARGUMENT_FUNC(T, fieldname, lossless_conversion)          \
  template <>                                                                  \
  T ProtoArgHelper::GetOptionalArg<T>(const std::string &arg_name,             \
                                      const T &default_value) const {          \
    if (arg_map_.count(arg_name) == 0) {                                       \
      VLOG(3) << "Using default parameter " << default_value << " for "        \
              << arg_name;                                                     \
      return default_value;                                                    \
    }                                                                          \
    CONDITIONS(arg_map_.at(arg_name).has_##fieldname(),                        \
               "Argument ",                                                    \
               arg_name,                                                       \
               " not found!");                                                 \
    auto value = arg_map_.at(arg_name).fieldname();                            \
    if (lossless_conversion) {                                                 \
      const bool castLossless = IsCastLossless<decltype(value), T>(value);     \
      CONDITIONS(castLossless,                                                 \
                 "Value",                                                      \
                 value,                                                        \
                 " of argument ",                                              \
                 arg_name,                                                     \
                 "cannot be casted losslessly to a target type");              \
    }                                                                          \
    return value;                                                              \
  }

GET_OPTIONAL_ARGUMENT_FUNC(float, f, false)
GET_OPTIONAL_ARGUMENT_FUNC(bool, i, false)
GET_OPTIONAL_ARGUMENT_FUNC(int, i, true)
GET_OPTIONAL_ARGUMENT_FUNC(std::string, s, false)
#undef GET_OPTIONAL_ARGUMENT_FUNC

#define GET_REPEATED_ARGUMENT_FUNC(T, fieldname, lossless_conversion)          \
  template <>                                                                  \
  std::vector<T> ProtoArgHelper::GetRepeatedArgs<T>(                           \
      const std::string &arg_name, const std::vector<T> &default_value)        \
      const {                                                                  \
    if (arg_map_.count(arg_name) == 0) {                                       \
      return default_value;                                                    \
    }                                                                          \
    std::vector<T> values;                                                     \
    for (const auto &v : arg_map_.at(arg_name).fieldname()) {                  \
      if (lossless_conversion) {                                               \
        const bool castLossless = IsCastLossless<decltype(v), T>(v);           \
        CONDITIONS(castLossless,                                               \
                   "Value",                                                    \
                   v,                                                          \
                   " of argument ",                                            \
                   arg_name,                                                   \
                   "cannot be casted losslessly to a target type");            \
      }                                                                        \
      values.push_back(v);                                                     \
    }                                                                          \
    return values;                                                             \
  }

GET_REPEATED_ARGUMENT_FUNC(float, floats, false)
GET_REPEATED_ARGUMENT_FUNC(int, ints, true)
GET_REPEATED_ARGUMENT_FUNC(int64_t, ints, true)
#undef GET_REPEATED_ARGUMENT_FUNC

bool IsQuantizedModel(const NetProto &net_def) {
  return ProtoArgHelper::GetOptionalArg<NetProto, int>(
             net_def, "quantize_flag", 0) == 1;
}

PruningType GetPruningType(const NetProto &net_def) {
  const int pruning_type_i = ProtoArgHelper::GetOptionalArg<NetProto, int>(
      net_def, "pruning_type", static_cast<PruningType>(PruningType::DENSE));
  return static_cast<PruningType>(pruning_type_i);
}

ModelType GetModelType(const NetProto &net_def) {
  const int model_type_i = ProtoArgHelper::GetOptionalArg<NetProto, int>(
      net_def, "model_type", static_cast<ModelType>(ModelType::DEFAULT));
  return static_cast<ModelType>(model_type_i);
}

} // namespace deepvan
