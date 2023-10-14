#ifndef DEEPVAN_BACKEND_OPS_REGISTRY_H_
#define DEEPVAN_BACKEND_OPS_REGISTRY_H_

#include "deepvan/core/operator.h"

namespace deepvan {
class OpRegistry : public OpRegistryBase {
 public:
  OpRegistry();
  ~OpRegistry() = default;
};
}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_OPS_REGISTRY_H_
