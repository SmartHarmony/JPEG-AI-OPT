#include <memory>

#include "deepvan/core/runtime/opencl/opencl_allocator.h"
#include "deepvan/core/runtime/opencl/opencl_runtime.h"

namespace deepvan {
namespace {

static cl_channel_type DataTypeToCLChannelType(const DataType t) {
  switch (t) {
    case DT_HALF:
      return CL_HALF_FLOAT;
    case DT_FLOAT:
      return CL_FLOAT;
    case DT_INT32:
      return CL_SIGNED_INT32;
    case DT_UINT8:
      return CL_UNSIGNED_INT32;
    case DT_INT8:
      return CL_SIGNED_INT8;
    default:
      LOG(FATAL) << "Image doesn't support the data type: " << t;
      return 0;
  }
}
}  // namespace

OpenCLAllocator::OpenCLAllocator(
    OpenCLRuntime *opencl_runtime):
    opencl_runtime_(opencl_runtime) {}

OpenCLAllocator::~OpenCLAllocator() {}
VanState OpenCLAllocator::New(size_t nbytes, void **result) const {
  if (nbytes == 0) {
    return VanState::SUCCEED;
  }
  VLOG(3) << "Allocate OpenCL buffer: " << nbytes;


  cl_int error;
  // TODO: check this flag's impact
  cl::Buffer *buffer = new cl::Buffer(opencl_runtime_->context(),
                                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      nbytes, nullptr, &error);
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL Buffer with "
                 << nbytes << " bytes failed because of "
                 << OpenCLErrorToString(error);
    delete buffer;
    *result = nullptr;
    return VanState::OUT_OF_RESOURCES;
  } else {
    *result = buffer;
    return VanState::SUCCEED;
  }
}

VanState OpenCLAllocator::NewImage(const std::vector<size_t> &image_shape,
                                     const DataType dt,
                                     void **result) const {
  CONDITIONS(image_shape.size() == 2, "Image shape's size must equal 2");
  VLOG(3) << "Allocate OpenCL image: " << image_shape[0] << ", "
          << image_shape[1];

  cl::ImageFormat img_format(CL_RGBA, DataTypeToCLChannelType(dt));

  cl_int error;
  cl::Image2D *cl_image =
      new cl::Image2D(opencl_runtime_->context(),
                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, img_format,
                      image_shape[0], image_shape[1], 0, nullptr, &error);
  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL image with shape: ["
                 << image_shape[0] << ", " << image_shape[1]
                 << "] failed because of "
                 << OpenCLErrorToString(error);
    delete cl_image;
    *result = nullptr;
    return VanState::OUT_OF_RESOURCES;
  } else {
    *result = cl_image;
    return VanState::SUCCEED;
  }
}

void OpenCLAllocator::Delete(void *buffer) const {
  VLOG(3) << "Free OpenCL buffer";
  if (buffer != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    delete cl_buffer;
  }
}

void OpenCLAllocator::DeleteImage(void *buffer) const {
  VLOG(3) << "Free OpenCL image";
  if (buffer != nullptr) {
    cl::Image2D *cl_image = static_cast<cl::Image2D *>(buffer);
    delete cl_image;
  }
}

void *OpenCLAllocator::Map(void *buffer, size_t offset, size_t nbytes) const {
  VLOG(3) << "Map OpenCL buffer";
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto queue = opencl_runtime_->command_queue();
  // TODO@vgod Non-blocking call
  cl_int error;
  void *mapped_ptr =
      queue.enqueueMapBuffer(*cl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                             offset, nbytes, nullptr, nullptr, &error);
  if (error != CL_SUCCESS) {
    LOG(ERROR) << "Map buffer failed, error: " << OpenCLErrorToString(error);
    mapped_ptr = nullptr;
  }
  return mapped_ptr;
}

// TODO@vgod there is something wrong with half type.
void *OpenCLAllocator::MapImage(void *buffer,
                                const std::vector<size_t> &image_shape,
                                std::vector<size_t> *mapped_image_pitch) const {
  VLOG(3) << "Map OpenCL Image";
  CONDITIONS(image_shape.size() == 2) << "Just support map 2d image";
  auto cl_image = static_cast<cl::Image2D *>(buffer);
  std::array<size_t, 3> origin = {0, 0, 0};
  std::array<size_t, 3> region = {image_shape[0], image_shape[1], 1};

  mapped_image_pitch->resize(2);
  cl_int error;
  void *mapped_ptr = opencl_runtime_->command_queue().enqueueMapImage(
      *cl_image, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
      mapped_image_pitch->data(), mapped_image_pitch->data() + 1, nullptr,
      nullptr, &error);
  if (error != CL_SUCCESS) {
    LOG(ERROR) << "Map Image failed, error: " << OpenCLErrorToString(error);
    mapped_ptr = nullptr;
  }
  return mapped_ptr;
}

void OpenCLAllocator::Unmap(void *buffer, void *mapped_ptr) const {
  VLOG(3) << "Unmap OpenCL buffer/Image";
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto queue = opencl_runtime_->command_queue();
  cl_int error = queue.enqueueUnmapMemObject(*cl_buffer, mapped_ptr,
                                             nullptr, nullptr);
  if (error != CL_SUCCESS) {
    LOG(ERROR) << "Unmap buffer failed, error: " << OpenCLErrorToString(error);
  }
}

bool OpenCLAllocator::OnHost() const { return false; }
}  // namespace deepvan
