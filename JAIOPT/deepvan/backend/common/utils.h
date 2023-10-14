#ifndef DEEPVAN_OPS_COMMON_UTILS_H_
#define DEEPVAN_OPS_COMMON_UTILS_H_

#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include "deepvan/core/types.h"

namespace deepvan {
namespace common {
namespace utils {

constexpr int64_t kTableSize = (1u << 10);

inline float CalculateResizeScale(index_t in_size,
                                  index_t out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size);
}

inline void GetFallbackTensor(const Tensor *&src,
                              OpContext *context,
                              bool cpu_allocator = false) {
  if (src != nullptr && !src->UnderlyingBuffer()->OnHost() &&
      src->dtype() == DataType::DT_HALF) {
    static int fallback_tensor_id = 0;
    std::string name = "";
    if (src->name().size() > 0) {
      name = "DeepVan-Fallback-Tensor-" + src->name();
    } else {
      name = "DeepVan-Fallback-Tensor-" + MakeString(fallback_tensor_id++);
    }

    NetworkController *ws = context->workspace();
    bool has_tensor = ws->HasTensor(name);
    if (has_tensor) {
      src = ws->GetTensor(name);
    } else {
      Tensor *fallbackTensor = ws->CreateTensor(
          name,
          cpu_allocator ? GetCPUAllocator() : context->device()->allocator(),
          DataType::DT_FLOAT);
      fallbackTensor->Resize(src->shape());
      Tensor::MappingGuard rhs_guard(src);
      Tensor::MappingGuard fallback_guard(fallbackTensor);
      const half *rhs_data = src->data<half>();
      float *fallback_data = fallbackTensor->mutable_data<float>();
#pragma omp parallel for schedule(runtime)
      for (int i = 0; i < src->size(); i++) {
        fallback_data[i] = half_float::half_cast<float>(rhs_data[i]);
      }
      src = fallbackTensor;
    }
  }
}

} // namespace utils
} // namespace common
} // namespace deepvan

#endif // DEEPVAN_OPS_COMMON_UTILS_H_
