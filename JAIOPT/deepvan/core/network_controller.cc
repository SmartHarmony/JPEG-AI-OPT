#include "deepvan/core/network_controller.h"

#include <unordered_set>
#include <utility>

#include "deepvan/core/arg_helper.h"
#include "deepvan/core/memory_optimizer.h"
// #include "deepvan/utils/quantize.h"

#ifdef OPENCL_SUPPORT
#include "deepvan/core/runtime/opencl/opencl_runtime.h"
#endif

namespace deepvan {
namespace {

bool HasHalfTensor(const NetProto &net_def) {
  for (auto &tensor : net_def.tensors()) {
    if (tensor.data_type() == DataType::DT_HALF) {
      return true;
    }
  }
  return false;
}

} // namespace

NetworkController::NetworkController() = default;

Tensor *NetworkController::CreateTensor(const std::string &name,
                                        Allocator *alloc,
                                        DataType type,
                                        bool is_weight) {
  if (HasTensor(name)) {
    VLOG(3) << "Tensor " << name << " already exists. Skipping.";
  } else {
    VLOG(3) << "Creating Tensor " << name;
    tensor_map_[name] =
        std::unique_ptr<Tensor>(new Tensor(alloc, type, is_weight, name));
  }
  return GetTensor(name);
}

const Tensor *NetworkController::GetTensor(const std::string &name) const {
  if (tensor_map_.count(name)) {
    return tensor_map_.at(name).get();
  } else {
    VLOG(1) << "Tensor " << name << " does not exist.";
  }
  return nullptr;
}

Tensor *NetworkController::GetTensor(const std::string &name) {
  return const_cast<Tensor *>(
      static_cast<const NetworkController *>(this)->GetTensor(name));
}

std::vector<std::string> NetworkController::Tensors() const {
  std::vector<std::string> names;
  for (auto &entry : tensor_map_) {
    names.push_back(entry.first);
  }
  return names;
}

VanState NetworkController::LoadModelTensor(const NetProto &net_def,
                                            Device *device,
                                            const unsigned char *model_data,
                                            const unsigned char *other_data) {
  LATENCY_LOGGER(1, "Load model tensors");
  index_t model_data_size = 0;
  index_t other_data_size = net_def.other_data_size();
  for (auto &const_tensor : net_def.tensors()) {
    int data_size = const_tensor.sparsed_weight()
                        ? const_tensor.data_size() +
                              const_tensor.col_index_data_size() +
                              const_tensor.row_ptr_data_size()
                        : const_tensor.data_size();
    model_data_size =
        std::max(model_data_size,
                 static_cast<index_t>(
                     const_tensor.offset() +
                     data_size * GetEnumTypeSize(const_tensor.data_type())));
  }
  VLOG(3) << "Model data size: " << model_data_size;
  VLOG(3) << "Other data size: " << other_data_size;
  const DeviceType device_type = device->device_type();

  if (model_data_size > 0) {
    // bool is_quantize_model = IsQuantizedModel(net_def);
    if (device_type == DeviceType::CPU) {
      tensor_buffer_ = std::unique_ptr<Buffer>(
          new Buffer(device->allocator(),
                     const_cast<unsigned char *>(model_data),
                     model_data_size));
      if (other_data_size > 0) {
        other_buffer_ = std::unique_ptr<Buffer>(
            new Buffer(device->allocator(),
                       const_cast<unsigned char *>(other_data),
                       other_data_size));
      }
    } else {
      tensor_buffer_ = std::unique_ptr<Buffer>(new Buffer(device->allocator()));
      RETURN_IF_ERROR(tensor_buffer_->Allocate(model_data_size));
      tensor_buffer_->Map(nullptr);
      tensor_buffer_->Copy(
          const_cast<unsigned char *>(model_data), 0, model_data_size);
      tensor_buffer_->UnMap();
    }

    for (auto &const_tensor : net_def.tensors()) {
      LATENCY_LOGGER(2, "Load tensor ", const_tensor.name());
      VLOG(3) << "Tensor name: " << const_tensor.name()
              << ", data type: " << const_tensor.data_type()
              << ", pattern: " << const_tensor.pattern_weight()
              << ", pruning_type: " << const_tensor.pruning_type()
              << ", shape: "
              << MakeString(std::vector<index_t>(const_tensor.dims().begin(),
                                                 const_tensor.dims().end()));
      std::vector<index_t> dims;
      for (const index_t d : const_tensor.dims()) {
        dims.push_back(d);
      }

      int data_size = const_tensor.data_size() +
                      const_tensor.row_ptr_data_size() +
                      const_tensor.col_index_data_size();
      std::unique_ptr<Tensor> tensor(new Tensor(
          BufferSlice(tensor_buffer_.get(),
                      const_tensor.offset(),
                      data_size * GetEnumTypeSize(const_tensor.data_type())),
          const_tensor.data_type(),
          true,
          const_tensor.name()));

      tensor->Reshape(dims);
      tensor->SetScale(const_tensor.scale());
      tensor->SetZeroPoint(const_tensor.zero_point());

      tensor_map_[const_tensor.name()] = std::move(tensor);
    }
  }
  return VanState::SUCCEED;
}

VanState NetworkController::PreallocateOutputTensor(
    const deepvan::NetProto &net_def,
    const deepvan::MemoryOptimizer *mem_optimizer,
    Device *device) {
  auto &mem_blocks = mem_optimizer->mem_blocks();
  for (auto &mem_block : mem_blocks) {
    VLOG(3) << "Preallocate memory block. id: " << mem_block.mem_id()
            << ", memory type: " << mem_block.mem_type()
            << ", size: " << mem_block.x() << "x" << mem_block.y();
    if (mem_block.mem_type() == MemoryType::CPU_BUFFER) {
      std::unique_ptr<BufferBase> tensor_buf(new Buffer(GetCPUAllocator()));
      RETURN_IF_ERROR(
          tensor_buf->Allocate(mem_block.x() + DEEPVAN_EXTRA_BUFFER_PAD_SIZE));
      preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                        std::move(tensor_buf));
    } else if (mem_block.mem_type() == MemoryType::GPU_IMAGE) {
      std::unique_ptr<BufferBase> image_buf(new Image(device->allocator()));
      RETURN_IF_ERROR(image_buf->Allocate({static_cast<size_t>(mem_block.x()),
                                           static_cast<size_t>(mem_block.y())},
                                          mem_block.data_type()));
      preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                        std::move(image_buf));
    } else if (mem_block.mem_type() == MemoryType::GPU_BUFFER) {
      std::unique_ptr<BufferBase> tensor_buf(new Buffer(device->allocator()));
      RETURN_IF_ERROR(
          tensor_buf->Allocate(mem_block.x() + DEEPVAN_EXTRA_BUFFER_PAD_SIZE));
      preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                        std::move(tensor_buf));
    }
  }
  VLOG(1) << "Preallocate buffer to tensors";
  bool is_quantize_model = IsQuantizedModel(net_def);
  for (auto &tensor_mem : mem_optimizer->tensor_mem_map()) {
    std::unique_ptr<Tensor> tensor(
        new Tensor(preallocated_allocator_.GetBuffer(tensor_mem.second.mem_id),
                   tensor_mem.second.data_type,
                   false,
                   tensor_mem.first));
    if (tensor_mem.second.has_data_format) {
      if (mem_blocks[tensor_mem.second.mem_id].mem_type() ==
          MemoryType::GPU_IMAGE) {
        VLOG(1) << "Tensor: " << tensor_mem.first
                << " Mem: " << tensor_mem.second.mem_id
                << " Data type: " << tensor->dtype()
                << " Image shape: " << tensor->UnderlyingBuffer()->shape()[0]
                << ", " << tensor->UnderlyingBuffer()->shape()[1];
        tensor->set_data_format(DataFormat::NHWC);
      } else {
        VLOG(1) << "Tensor: " << tensor_mem.first
                << " Mem: " << tensor_mem.second.mem_id
                << " Data type: " << tensor->dtype()
                << ", Buffer size: " << tensor->UnderlyingBuffer()->size();
        if (mem_blocks[tensor_mem.second.mem_id].mem_type() ==
            MemoryType::GPU_BUFFER
            // ||is_quantize_model
        ) {
          tensor->set_data_format(DataFormat::NHWC);
        } else {
          tensor->set_data_format(DataFormat::NCHW);
        }
      }
    } else {
      tensor->set_data_format(DataFormat::DF_NONE);
    }
    tensor_map_[tensor_mem.first] = std::move(tensor);
  }

   return VanState::SUCCEED;
}

void NetworkController::RemoveUnusedBuffer() {
  auto iter = tensor_map_.begin();
  auto end_iter = tensor_map_.end();
  while (iter != end_iter) {
    auto old_iter = iter++;
    if (old_iter->second->unused()) {
      tensor_map_.erase(old_iter);
    }
  }
  tensor_buffer_.reset(nullptr);
}

void NetworkController::RemoveAllBuffer() {
  auto iter = tensor_map_.begin();
  auto end_iter = tensor_map_.end();
  while (iter != end_iter) {
    auto old_iter = iter++;
    tensor_map_.erase(old_iter);
  }
  tensor_buffer_.reset(nullptr);
  other_buffer_.reset(nullptr);
}

void NetworkController::RemoveAndReloadBuffer(const NetProto &net_def,
                                              const unsigned char *model_data,
                                              Allocator *alloc) {
  std::unordered_set<std::string> tensor_to_host;
  for (auto &op : net_def.op()) {
    if (op.device_type() == DeviceType::CPU) {
      for (std::string input : op.input()) {
        tensor_to_host.insert(input);
      }
    }
  }
  for (auto &const_tensor : net_def.tensors()) {
    auto iter = tensor_map_.find(const_tensor.name());
    if (iter->second->unused()) {
      tensor_map_.erase(iter);
    } else {
      std::vector<index_t> dims;
      for (const index_t d : const_tensor.dims()) {
        dims.push_back(d);
      }

      if (tensor_to_host.find(const_tensor.name()) != tensor_to_host.end() &&
          const_tensor.data_type() == DataType::DT_HALF) {
        std::unique_ptr<Tensor> tensor(
            new Tensor(alloc, DataType::DT_FLOAT, true, const_tensor.name()));
        tensor->Resize(dims);
        CONDITIONS(tensor->size() == const_tensor.data_size(),
                   "Tensor's data_size not equal with the shape");
        Tensor::MappingGuard guard(tensor.get());
        float *dst_data = tensor->mutable_data<float>();
        const half *org_data =
            reinterpret_cast<const half *>(model_data + const_tensor.offset());
        for (index_t i = 0; i < const_tensor.data_size(); ++i) {
          dst_data[i] = half_float::half_cast<float>(org_data[i]);
        }
        tensor_map_[const_tensor.name()] = std::move(tensor);
      } else {
        std::unique_ptr<Tensor> tensor(new Tensor(
            alloc, const_tensor.data_type(), true, const_tensor.name()));
        tensor->Resize(dims);
        CONDITIONS(tensor->size() == const_tensor.data_size(),
                   "Tensor's data_size not equal with the shape");
        tensor->CopyBytes(model_data + const_tensor.offset(),
                          const_tensor.data_size() *
                              GetEnumTypeSize(const_tensor.data_type()));
        tensor_map_[const_tensor.name()] = std::move(tensor);
      }
    }
  }
  tensor_buffer_.reset(nullptr);
}

void NetworkController::RemoveTensor(const std::string &name) {
  auto iter = tensor_map_.find(name);
  if (iter != tensor_map_.end()) {
    tensor_map_.erase(iter);
  }
}
} // namespace deepvan
