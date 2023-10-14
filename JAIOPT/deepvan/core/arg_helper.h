#ifndef DEEPVAN_CORE_ARG_HELPER_H_
#define DEEPVAN_CORE_ARG_HELPER_H_

#include <map>
#include <string>
#include <vector>

#include "deepvan/proto/deepvan.pb.h"

namespace deepvan {
// Refer to caffe2
class ProtoArgHelper {
public:
  template <typename Def, typename T>
  static T GetOptionalArg(const Def &def,
                          const std::string &arg_name,
                          const T &default_value) {
    return ProtoArgHelper(def).GetOptionalArg<T>(arg_name, default_value);
  }

  template <typename Def, typename T>
  static std::vector<T>
  GetRepeatedArgs(const Def &def,
                  const std::string &arg_name,
                  const std::vector<T> &default_value = std::vector<T>()) {
    return ProtoArgHelper(def).GetRepeatedArgs<T>(arg_name, default_value);
  }

  explicit ProtoArgHelper(const OperatorProto &def);
  explicit ProtoArgHelper(const NetProto &NetProto);

  template <typename T>
  T GetOptionalArg(const std::string &arg_name, const T &default_value) const;
  template <typename T>
  std::vector<T>
  GetRepeatedArgs(const std::string &arg_name,
                  const std::vector<T> &default_value = std::vector<T>()) const;

private:
  std::map<std::string, ArgumentProto> arg_map_;
};

bool IsQuantizedModel(const NetProto &def);

PruningType GetPruningType(const NetProto &net_def);

ModelType GetModelType(const NetProto &net_def);
} // namespace deepvan

#endif // DEEPVAN_CORE_ARG_HELPER_H_
