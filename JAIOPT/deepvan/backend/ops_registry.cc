#include "deepvan/backend/ops_registry.h"

namespace deepvan {

// Keep in lexicographical order
extern void RegisterActivation(OpRegistryBase *op_registry);
extern void RegisterBufferTransform(OpRegistryBase *op_registry);
extern void RegisterConv2D(OpRegistryBase *op_registry);
extern void RegisterDeconv2D(OpRegistryBase *op_registry);
extern void RegisterEltwise(OpRegistryBase *op_registry);

OpRegistry::OpRegistry() : OpRegistryBase() {
  // Keep in lexicographical order
  RegisterActivation(this);
  RegisterBufferTransform(this);
  RegisterDeconv2D(this);
  RegisterConv2D(this);
  RegisterEltwise(this);
}
} // namespace deepvan
