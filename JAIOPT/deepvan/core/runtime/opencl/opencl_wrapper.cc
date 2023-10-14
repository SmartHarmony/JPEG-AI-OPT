#include <CL/opencl.h>
#include <dlfcn.h>
#include <string>
#include <vector>

#include "deepvan/compat/port-arch.h"
#include "deepvan/utils/logging.h"

#ifdef MEMPROF_SUPPORT
// When memprof is defined, this file should not intercept all open cl invocations
// Otherwise, memprof cannot intercept GPU memory usages.
#include "deepvan/tools/libmemprofile/memory_usage.h"
#endif // MEMPROF_SUPPORT

/**
 * Wrapper of OpenCL 2.0, based on file opencl20/CL/cl.h
 */

#ifdef DEEPVAN_OS_MAC
typedef cl_queue_properties_APPLE cl_queue_properties;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION < 200
#define CL_API_SUFFIX__VERSION_2_0
#endif

namespace deepvan {
namespace runtime {

class OpenCLLibrary final {
 private:
  OpenCLLibrary();
  DISABLE_COPY_AND_ASSIGN(OpenCLLibrary);

  bool Load();
  void *LoadFromPath(const std::string &path);

 public:
  static OpenCLLibrary *Get();

  using clGetPlatformIDsFunc = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
  using clGetPlatformInfoFunc =
      cl_int (*)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
  using clBuildProgramFunc = cl_int (*)(cl_program,
                                        cl_uint,
                                        const cl_device_id *,
                                        const char *,
                                        void (*pfn_notify)(cl_program, void *),
                                        void *);
  using clEnqueueNDRangeKernelFunc = cl_int (*)(cl_command_queue,
                                                cl_kernel,
                                                cl_uint,
                                                const size_t *,
                                                const size_t *,
                                                const size_t *,
                                                cl_uint,
                                                const cl_event *,
                                                cl_event *);
  using clSetKernelArgFunc = cl_int (*)(cl_kernel,
                                        cl_uint,
                                        size_t,
                                        const void *);
  using clRetainMemObjectFunc = cl_int (*)(cl_mem);
  using clReleaseMemObjectFunc = cl_int (*)(cl_mem);
  using clEnqueueUnmapMemObjectFunc = cl_int (*)(
      cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
  using clRetainCommandQueueFunc = cl_int (*)(cl_command_queue command_queue);
  using clCreateContextFunc =
      cl_context (*)(const cl_context_properties *,
                     cl_uint,
                     const cl_device_id *,
                     void(CL_CALLBACK *)(  // NOLINT(readability/casting)
                         const char *, const void *, size_t, void *),
                     void *,
                     cl_int *);
  using clCreateContextFromTypeFunc =
      cl_context (*)(const cl_context_properties *,
                     cl_device_type,
                     void(CL_CALLBACK *)(  // NOLINT(readability/casting)
                         const char *, const void *, size_t, void *),
                     void *,
                     cl_int *);
  using clReleaseContextFunc = cl_int (*)(cl_context);
  using clWaitForEventsFunc = cl_int (*)(cl_uint, const cl_event *);
  using clReleaseEventFunc = cl_int (*)(cl_event);
  using clEnqueueWriteBufferFunc = cl_int (*)(cl_command_queue,
                                              cl_mem,
                                              cl_bool,
                                              size_t,
                                              size_t,
                                              const void *,
                                              cl_uint,
                                              const cl_event *,
                                              cl_event *);
  using clEnqueueReadBufferFunc = cl_int (*)(cl_command_queue,
                                             cl_mem,
                                             cl_bool,
                                             size_t,
                                             size_t,
                                             void *,
                                             cl_uint,
                                             const cl_event *,
                                             cl_event *);
  using clGetProgramBuildInfoFunc = cl_int (*)(cl_program,
                                               cl_device_id,
                                               cl_program_build_info,
                                               size_t,
                                               void *,
                                               size_t *);
  using clRetainProgramFunc = cl_int (*)(cl_program program);
  using clEnqueueMapBufferFunc = void *(*)(cl_command_queue,
                                           cl_mem,
                                           cl_bool,
                                           cl_map_flags,
                                           size_t,
                                           size_t,
                                           cl_uint,
                                           const cl_event *,
                                           cl_event *,
                                           cl_int *);
  using clEnqueueMapImageFunc = void *(*)(cl_command_queue,
                                          cl_mem,
                                          cl_bool,
                                          cl_map_flags,
                                          const size_t *,
                                          const size_t *,
                                          size_t *,
                                          size_t *,
                                          cl_uint,
                                          const cl_event *,
                                          cl_event *,
                                          cl_int *);
  using clCreateCommandQueueFunc = cl_command_queue(CL_API_CALL *)(  // NOLINT
      cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
  using clCreateCommandQueueWithPropertiesFunc = cl_command_queue (*)(
      cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
  using clReleaseCommandQueueFunc = cl_int (*)(cl_command_queue);
  using clCreateProgramWithBinaryFunc = cl_program (*)(cl_context,
                                                       cl_uint,
                                                       const cl_device_id *,
                                                       const size_t *,
                                                       const unsigned char **,
                                                       cl_int *,
                                                       cl_int *);
  using clRetainContextFunc = cl_int (*)(cl_context context);
  using clGetContextInfoFunc =
      cl_int (*)(cl_context, cl_context_info, size_t, void *, size_t *);
  using clReleaseProgramFunc = cl_int (*)(cl_program program);
  using clFlushFunc = cl_int (*)(cl_command_queue command_queue);
  using clFinishFunc = cl_int (*)(cl_command_queue command_queue);
  using clGetProgramInfoFunc =
      cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *);
  using clCreateKernelFunc = cl_kernel (*)(cl_program, const char *, cl_int *);
  using clRetainKernelFunc = cl_int (*)(cl_kernel kernel);
  using clCreateBufferFunc =
      cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
  using clCreateImage2DFunc = cl_mem(CL_API_CALL *)(cl_context,  // NOLINT
                                                    cl_mem_flags,
                                                    const cl_image_format *,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    void *,
                                                    cl_int *);
  using clGetMemObjectInfoFunc = cl_int (*)(cl_mem,
					cl_mem_info,
					size_t,
 					void*,
  					size_t*);
  using clCreateImageFunc = cl_mem (*)(cl_context,
                                       cl_mem_flags,
                                       const cl_image_format *,
                                       const cl_image_desc *,
                                       void *,
                                       cl_int *);
  using clCreateProgramWithSourceFunc = cl_program (*)(
      cl_context, cl_uint, const char **, const size_t *, cl_int *);
  using clReleaseKernelFunc = cl_int (*)(cl_kernel kernel);
  using clGetDeviceInfoFunc =
      cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
  using clGetDeviceIDsFunc = cl_int (*)(
      cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
  using clRetainDeviceFunc = cl_int (*)(cl_device_id);
  using clReleaseDeviceFunc = cl_int (*)(cl_device_id);
  using clRetainEventFunc = cl_int (*)(cl_event);
  using clGetKernelWorkGroupInfoFunc = cl_int (*)(cl_kernel,
                                                  cl_device_id,
                                                  cl_kernel_work_group_info,
                                                  size_t,
                                                  void *,
                                                  size_t *);
  using clGetEventInfoFunc = cl_int (*)(cl_event event,
                                        cl_event_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret);
  using clGetEventProfilingInfoFunc = cl_int (*)(cl_event event,
                                                 cl_profiling_info param_name,
                                                 size_t param_value_size,
                                                 void *param_value,
                                                 size_t *param_value_size_ret);
  using clGetImageInfoFunc =
      cl_int (*)(cl_mem, cl_image_info, size_t, void *, size_t *);

#define CL_DEFINE_FUNC_PTR(func) func##Func func = nullptr

  CL_DEFINE_FUNC_PTR(clGetPlatformIDs);
  CL_DEFINE_FUNC_PTR(clGetPlatformInfo);
  CL_DEFINE_FUNC_PTR(clBuildProgram);
  CL_DEFINE_FUNC_PTR(clEnqueueNDRangeKernel);
  CL_DEFINE_FUNC_PTR(clSetKernelArg);
  CL_DEFINE_FUNC_PTR(clReleaseKernel);
  CL_DEFINE_FUNC_PTR(clCreateProgramWithSource);
  CL_DEFINE_FUNC_PTR(clCreateBuffer);
  CL_DEFINE_FUNC_PTR(clCreateImage);
  CL_DEFINE_FUNC_PTR(clCreateImage2D);
  CL_DEFINE_FUNC_PTR(clReleaseMemObject);
  CL_DEFINE_FUNC_PTR(clRetainKernel);
  CL_DEFINE_FUNC_PTR(clCreateKernel);
  CL_DEFINE_FUNC_PTR(clGetProgramInfo);
  CL_DEFINE_FUNC_PTR(clFlush);
  CL_DEFINE_FUNC_PTR(clFinish);
  CL_DEFINE_FUNC_PTR(clReleaseProgram);
  CL_DEFINE_FUNC_PTR(clRetainContext);
  CL_DEFINE_FUNC_PTR(clGetContextInfo);
  CL_DEFINE_FUNC_PTR(clCreateProgramWithBinary);
  CL_DEFINE_FUNC_PTR(clCreateCommandQueue);
  CL_DEFINE_FUNC_PTR(clCreateCommandQueueWithProperties);
  CL_DEFINE_FUNC_PTR(clReleaseCommandQueue);
  CL_DEFINE_FUNC_PTR(clEnqueueMapBuffer);
  CL_DEFINE_FUNC_PTR(clEnqueueMapImage);
  CL_DEFINE_FUNC_PTR(clRetainProgram);
  CL_DEFINE_FUNC_PTR(clGetProgramBuildInfo);
  CL_DEFINE_FUNC_PTR(clEnqueueReadBuffer);
  CL_DEFINE_FUNC_PTR(clEnqueueWriteBuffer);
  CL_DEFINE_FUNC_PTR(clWaitForEvents);
  CL_DEFINE_FUNC_PTR(clReleaseEvent);
  CL_DEFINE_FUNC_PTR(clCreateContext);
  CL_DEFINE_FUNC_PTR(clCreateContextFromType);
  CL_DEFINE_FUNC_PTR(clReleaseContext);
  CL_DEFINE_FUNC_PTR(clRetainCommandQueue);
  CL_DEFINE_FUNC_PTR(clEnqueueUnmapMemObject);
  CL_DEFINE_FUNC_PTR(clRetainMemObject);
  CL_DEFINE_FUNC_PTR(clGetMemObjectInfo);
  CL_DEFINE_FUNC_PTR(clGetDeviceInfo);
  CL_DEFINE_FUNC_PTR(clGetDeviceIDs);
  CL_DEFINE_FUNC_PTR(clRetainDevice);
  CL_DEFINE_FUNC_PTR(clReleaseDevice);
  CL_DEFINE_FUNC_PTR(clRetainEvent);
  CL_DEFINE_FUNC_PTR(clGetKernelWorkGroupInfo);
  CL_DEFINE_FUNC_PTR(clGetEventInfo);
  CL_DEFINE_FUNC_PTR(clGetEventProfilingInfo);
  CL_DEFINE_FUNC_PTR(clGetImageInfo);

#undef CL_DEFINE_FUNC_PTR

 private:
  void *handle_ = nullptr;
};

OpenCLLibrary *OpenCLLibrary::Get() {
  static OpenCLLibrary library;
  return &library;
}

OpenCLLibrary::OpenCLLibrary() {
  this->Load();
  // Do not call dlclose which may unload all OpenCL symbols.
  // If close the OpenCL library, the static OpenCLRuntime destructor may fail.
  // If there is no dlclose, the library will be closed when the program exist.
  // Besides, the library will not be load repeatedly even dlopen many times.
}

bool OpenCLLibrary::Load() {
  if (handle_ != nullptr) {
    return true;
  }

  // Add customized OpenCL search path here
  const std::vector<std::string> paths = {
    "libOpenCL.so",
#if defined(__aarch64__)
    // Qualcomm Adreno with Android
    "/system/vendor/lib64/libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    // Android 12
    "/vendor/lib64/libOpenCL.so",
    "/vendor/lib64/libOpenCL-pixel.so",
    "libOpenCL-pixel.so",
    // Mali with Android
    "/system/vendor/lib64/egl/libGLES_mali.so",
    "/system/lib64/egl/libGLES_mali.so",
    // Typical Linux board
    "/usr/lib/aarch64-linux-gnu/libOpenCL.so",
#else
    // Qualcomm Adreno with Android
    "/system/vendor/lib/libOpenCL.so",
    "/system/lib/libOpenCL.so",
    // Android 12
    "/vendor/lib/libOpenCL.so",
    "/vendor/lib/libOpenCL-pixel.so",
    "libOpenCL-pixel.so",
    // Mali with Android
    "/system/vendor/lib/egl/libGLES_mali.so",
    "/system/lib/egl/libGLES_mali.so",
    // Typical Linux board
    "/usr/lib/arm-linux-gnueabihf/libOpenCL.so",
#endif
  };

  for (const auto &path : paths) {
    VLOG(2) << "Loading OpenCL from " << path;
    void *handle = LoadFromPath(path);
    if (handle != nullptr) {
      handle_ = handle;
      break;
    }
  }

  if (handle_ == nullptr) {
    LOG(ERROR) << "Failed to load OpenCL library, "
        "please make sure there exists OpenCL library on your device, "
        "and your APP have right to access the library.";
    return false;
  }

  return true;
}

void *OpenCLLibrary::LoadFromPath(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (handle == nullptr) {
    VLOG(2) << "Failed to load OpenCL library from path " << path
            << " error code: " << dlerror();
    return nullptr;
  }

#define CL_ASSIGN_FROM_DLSYM(func)                          \
  do {                                                           \
    void *ptr = dlsym(handle, #func);                            \
    if (ptr == nullptr) {                                        \
      VLOG(1) << "Failed to load " << #func << " from " << path; \
      continue;                                                  \
    }                                                            \
    func = reinterpret_cast<func##Func>(ptr);                    \
    VLOG(2) << "Loaded " << #func << " from " << path;           \
  } while (false)

  CL_ASSIGN_FROM_DLSYM(clGetPlatformIDs);
  CL_ASSIGN_FROM_DLSYM(clGetPlatformInfo);
  CL_ASSIGN_FROM_DLSYM(clBuildProgram);
  CL_ASSIGN_FROM_DLSYM(clEnqueueNDRangeKernel);
  CL_ASSIGN_FROM_DLSYM(clSetKernelArg);
  CL_ASSIGN_FROM_DLSYM(clReleaseKernel);
  CL_ASSIGN_FROM_DLSYM(clCreateProgramWithSource);


  CL_ASSIGN_FROM_DLSYM(clCreateBuffer);
  CL_ASSIGN_FROM_DLSYM(clCreateImage);
  CL_ASSIGN_FROM_DLSYM(clCreateImage2D);
  CL_ASSIGN_FROM_DLSYM(clReleaseMemObject);
  CL_ASSIGN_FROM_DLSYM(clRetainKernel);
  CL_ASSIGN_FROM_DLSYM(clCreateKernel);
  CL_ASSIGN_FROM_DLSYM(clGetProgramInfo);
  CL_ASSIGN_FROM_DLSYM(clFlush);
  CL_ASSIGN_FROM_DLSYM(clFinish);
  CL_ASSIGN_FROM_DLSYM(clReleaseProgram);
  CL_ASSIGN_FROM_DLSYM(clRetainContext);
  CL_ASSIGN_FROM_DLSYM(clGetContextInfo);
  CL_ASSIGN_FROM_DLSYM(clCreateProgramWithBinary);
  CL_ASSIGN_FROM_DLSYM(clCreateCommandQueue);
  CL_ASSIGN_FROM_DLSYM(clCreateCommandQueueWithProperties);
  CL_ASSIGN_FROM_DLSYM(clReleaseCommandQueue);
  CL_ASSIGN_FROM_DLSYM(clEnqueueMapBuffer);
  CL_ASSIGN_FROM_DLSYM(clEnqueueMapImage);
  CL_ASSIGN_FROM_DLSYM(clRetainProgram);
  CL_ASSIGN_FROM_DLSYM(clGetProgramBuildInfo);
  CL_ASSIGN_FROM_DLSYM(clEnqueueReadBuffer);
  CL_ASSIGN_FROM_DLSYM(clEnqueueWriteBuffer);
  CL_ASSIGN_FROM_DLSYM(clWaitForEvents);
  CL_ASSIGN_FROM_DLSYM(clReleaseEvent);
  CL_ASSIGN_FROM_DLSYM(clCreateContext);
  CL_ASSIGN_FROM_DLSYM(clCreateContextFromType);
  CL_ASSIGN_FROM_DLSYM(clReleaseContext);
  CL_ASSIGN_FROM_DLSYM(clRetainCommandQueue);
  CL_ASSIGN_FROM_DLSYM(clEnqueueUnmapMemObject);
  CL_ASSIGN_FROM_DLSYM(clRetainMemObject);
  CL_ASSIGN_FROM_DLSYM(clGetMemObjectInfo);
  CL_ASSIGN_FROM_DLSYM(clGetDeviceInfo);
  CL_ASSIGN_FROM_DLSYM(clGetDeviceIDs);
  CL_ASSIGN_FROM_DLSYM(clRetainDevice);
  CL_ASSIGN_FROM_DLSYM(clReleaseDevice);
  CL_ASSIGN_FROM_DLSYM(clRetainEvent);
  CL_ASSIGN_FROM_DLSYM(clGetKernelWorkGroupInfo);
  CL_ASSIGN_FROM_DLSYM(clGetEventInfo);
  CL_ASSIGN_FROM_DLSYM(clGetEventProfilingInfo);
  CL_ASSIGN_FROM_DLSYM(clGetImageInfo);

#undef CL_ASSIGN_FROM_DLSYM

  return handle;
}

}  // namespace runtime
}  // namespace deepvan

#ifdef MEMPROF_SUPPORT
size_t getClMemSize(cl_mem memobj) {
  size_t size = 0;
  
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetMemObjectInfo;
  
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetMemObjectInfo");
    if(func(memobj, CL_MEM_SIZE, sizeof(size), &size, NULL) == CL_SUCCESS) {
      return size; 
    }
  } 
    
  return 0;
}
#endif

// Platform APIs
CL_API_ENTRY cl_int clGetPlatformIDs(cl_uint num_entries,
                                     cl_platform_id *platforms,
                                     cl_uint *num_platforms)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetPlatformIDs;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetPlatformIDs");
    return func(num_entries, platforms, num_platforms);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetPlatformInfo(cl_platform_id platform,
                                      cl_platform_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetPlatformInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetPlatformInfo");
    return func(platform, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Device APIs
CL_API_ENTRY cl_int clGetDeviceIDs(cl_platform_id platform,
                                   cl_device_type device_type,
                                   cl_uint num_entries,
                                   cl_device_id *devices,
                                   cl_uint *num_devices)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetDeviceIDs;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetDeviceIDs");
    return func(platform, device_type, num_entries, devices, num_devices);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetDeviceInfo(cl_device_id device,
                                    cl_device_info param_name,
                                    size_t param_value_size,
                                    void *param_value,
                                    size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetDeviceInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetDeviceInfo");
    return func(device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clRetainDevice(cl_device_id device)
    CL_API_SUFFIX__VERSION_1_2 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clRetainDevice;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clRetainDevice");
    return func(device);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseDevice(cl_device_id device)
    CL_API_SUFFIX__VERSION_1_2 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clReleaseDevice;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clReleaseDevice");
    return func(device);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Context APIs
CL_API_ENTRY cl_context clCreateContext(
    const cl_context_properties *properties,
    cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateContext;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateContext");
    return func(properties, num_devices, devices, pfn_notify, user_data,
                errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_context clCreateContextFromType(
    const cl_context_properties *properties,
    cl_device_type device_type,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateContextFromType;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateContextFromType");
    return func(properties, device_type, pfn_notify, user_data, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clRetainContext(cl_context context)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clRetainContext;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clRetainContext");
    return func(context);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseContext(cl_context context)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clReleaseContext;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clReleaseContext");
    return func(context);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetContextInfo(cl_context context,
                                     cl_context_info param_name,
                                     size_t param_value_size,
                                     void *param_value,
                                     size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetContextInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetContextInfo");
    return func(context, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Program Object APIs
CL_API_ENTRY cl_program clCreateProgramWithSource(cl_context context,
                                                  cl_uint count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateProgramWithSource;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateProgramWithSource");
    cl_program object = func(context, count, strings, lengths, errcode_ret);

  #ifdef MEMPROF_SUPPORT
    if(errcode_ret == CL_SUCCESS) {
      size_t deviceCount;
      size_t sizeArray[8]; // No need to use malloc here, as the number of device is typically very small
      auto getProgramInfo = deepvan::runtime::OpenCLLibrary::Get()->clGetProgramInfo;
      getProgramInfo(object, CL_PROGRAM_NUM_DEVICES, sizeof(deviceCount), &deviceCount, NULL);

      getProgramInfo(object, CL_PROGRAM_BINARY_SIZES, sizeof(sizeArray), sizeArray, NULL);
      size_t size = 0;

      for(size_t i = 0; i < deviceCount; i++) {
      	size += sizeArray[i];
      }
      memory_usage::getInstance().updateGPUAlloc((void *)object, size);
   }
   #endif
    return object;

  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_program
clCreateProgramWithBinary(cl_context context,
                          cl_uint num_devices,
                          const cl_device_id *device_list,
                          const size_t *lengths,
                          const unsigned char **binaries,
                          cl_int *binary_status,
                          cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateProgramWithBinary;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateProgramWithBinary");
    return func(context, num_devices, device_list, lengths, binaries,
                binary_status, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clGetProgramInfo(cl_program program,
                                     cl_program_info param_name,
                                     size_t param_value_size,
                                     void *param_value,
                                     size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetProgramInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetProgramInfo");
    return func(program, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetProgramBuildInfo(cl_program program,
                                          cl_device_id device,
                                          cl_program_build_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetProgramBuildInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetProgramBuildInfo");
    return func(program, device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clRetainProgram(cl_program program)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clRetainProgram;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clRetainProgram");
    return func(program);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseProgram(cl_program program)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clReleaseProgram;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clReleaseProgram");
    return func(program);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clBuildProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const char *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clBuildProgram;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clBuildProgram");
    return func(program, num_devices, device_list, options, pfn_notify,
                user_data);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Kernel Object APIs
CL_API_ENTRY cl_kernel clCreateKernel(cl_program program,
                                      const char *kernel_name,
                                      cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateKernel;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateKernel");
    return func(program, kernel_name, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clRetainKernel(cl_kernel kernel)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clRetainKernel;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clRetainKernel");
    return func(kernel);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseKernel(cl_kernel kernel)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clReleaseKernel;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clReleaseKernel");
    return func(kernel);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clSetKernelArg(cl_kernel kernel,
                                   cl_uint arg_index,
                                   size_t arg_size,
                                   const void *arg_value)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clSetKernelArg;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clSetKernelArg");
    return func(kernel, arg_index, arg_size, arg_value);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Memory Object APIs
CL_API_ENTRY cl_mem clCreateBuffer(cl_context context,
                                   cl_mem_flags flags,
                                   size_t size,
                                   void *host_ptr,
                                   cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateBuffer;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateBuffer");
    cl_mem object = func(context, flags, size, host_ptr, errcode_ret);
 #ifdef MEMPROF_SUPPORT
    if(object != NULL) {
      memory_usage::getInstance().updateGPUAlloc((void *)object, getClMemSize(object));
    }
 #endif
    return object;
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_mem clCreateImage(cl_context context,
                                  cl_mem_flags flags,
                                  const cl_image_format *image_format,
                                  const cl_image_desc *image_desc,
                                  void *host_ptr,
                                  cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_2 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateImage;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateImage");
    cl_mem object = func(context,
                         flags,
                         image_format,
                         image_desc,
                         host_ptr,
                         errcode_ret);
  #ifdef MEMPROF_SUPPORT
    if(object != NULL) {
      memory_usage::getInstance().updateGPUAlloc((void *)object, getClMemSize(object));
    }
  #endif   
    return object;
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clRetainMemObject(cl_mem memobj)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clRetainMemObject;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clRetainMemObject");
    return func(memobj);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseMemObject(cl_mem memobj)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clReleaseMemObject;
  if (func != nullptr) {
  #ifdef MEMPROF_SUPPORT
    if(memobj != NULL) {
      memory_usage::getInstance().updateGPUFree((void *)memobj, getClMemSize(memobj));
    }
  #endif
    LATENCY_LOGGER(3, "clReleaseMemObject");
    return func(memobj);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetImageInfo(cl_mem image,
                                   cl_image_info param_name,
                                   size_t param_value_size,
                                   void *param_value,
                                   size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetImageInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetImageInfo");
    return func(image, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Command Queue APIs
CL_API_ENTRY cl_command_queue clCreateCommandQueueWithProperties(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties *properties,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0 {
  auto func =
      deepvan::runtime::OpenCLLibrary::Get()->clCreateCommandQueueWithProperties;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateCommandQueueWithProperties");
    return func(context, device, properties, errcode_ret);
  } else {
    // Fix MediaTek MT6771 OpenCL driver breakage
    VLOG(2) << "Fallback to clCreateCommandQueue";
    if (properties[0] == CL_QUEUE_PROPERTIES) {
      // When calling with OpenCL-CLHPP, the 2nd param is provided by caller.
#pragma GCC diagnostic push  // disable warning both for clang and gcc
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      return clCreateCommandQueue(context, device, properties[1], errcode_ret);
#pragma GCC diagnostic pop
    } else {
      LOG(FATAL) << "Unknown calling parameters, check the code here";
      if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
      return nullptr;
    }
  }
}

CL_API_ENTRY cl_int clRetainCommandQueue(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clRetainCommandQueue;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clRetainCommandQueue");
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseCommandQueue(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clReleaseCommandQueue;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clReleaseCommandQueue");
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Enqueued Commands APIs
CL_API_ENTRY cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                                        cl_mem buffer,
                                        cl_bool blocking_read,
                                        size_t offset,
                                        size_t size,
                                        void *ptr,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clEnqueueReadBuffer;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clEnqueueReadBuffer");
    return func(command_queue, buffer, blocking_read, offset, size, ptr,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
                                         cl_mem buffer,
                                         cl_bool blocking_write,
                                         size_t offset,
                                         size_t size,
                                         const void *ptr,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event *event_wait_list,
                                         cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clEnqueueWriteBuffer;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clEnqueueWriteBuffer");
    return func(command_queue, buffer, blocking_write, offset, size, ptr,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY void *clEnqueueMapBuffer(cl_command_queue command_queue,
                                      cl_mem buffer,
                                      cl_bool blocking_map,
                                      cl_map_flags map_flags,
                                      size_t offset,
                                      size_t size,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event *event_wait_list,
                                      cl_event *event,
                                      cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clEnqueueMapBuffer;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clEnqueueMapBuffer");
    return func(command_queue, buffer, blocking_map, map_flags, offset, size,
                num_events_in_wait_list, event_wait_list, event, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY void *clEnqueueMapImage(cl_command_queue command_queue,
                                     cl_mem image,
                                     cl_bool blocking_map,
                                     cl_map_flags map_flags,
                                     const size_t *origin,
                                     const size_t *region,
                                     size_t *image_row_pitch,
                                     size_t *image_slice_pitch,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event *event_wait_list,
                                     cl_event *event,
                                     cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clEnqueueMapImage;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clEnqueueMapImage");
    return func(command_queue, image, blocking_map, map_flags, origin, region,
                image_row_pitch, image_slice_pitch, num_events_in_wait_list,
                event_wait_list, event, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                                            cl_mem memobj,
                                            void *mapped_ptr,
                                            cl_uint num_events_in_wait_list,
                                            const cl_event *event_wait_list,
                                            cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clEnqueueUnmapMemObject;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clEnqueueUnmapMemObject");
    return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetKernelWorkGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetKernelWorkGroupInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetKernelWorkGroupInfo");
    return func(kernel, device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
                                           cl_kernel kernel,
                                           cl_uint work_dim,
                                           const size_t *global_work_offset,
                                           const size_t *global_work_size,
                                           const size_t *local_work_size,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event *event_wait_list,
                                           cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clEnqueueNDRangeKernel;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clEnqueueNDRangeKernel");
    return func(command_queue, kernel, work_dim, global_work_offset,
                global_work_size, local_work_size, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Event Object APIs
CL_API_ENTRY cl_int clWaitForEvents(
    cl_uint num_events, const cl_event *event_list) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clWaitForEvents;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clWaitForEvents");
    return func(num_events, event_list);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clRetainEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clRetainEvent;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clRetainEvent");
    return func(event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clReleaseEvent;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clReleaseEvent");
    return func(event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Event API
CL_API_ENTRY cl_int clGetEventInfo(cl_event event,
                                   cl_event_info param_name,
                                   size_t param_value_size,
                                   void *param_value,
                                   size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetEventInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetEventInfo");
    return func(event, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Profiling APIs
CL_API_ENTRY cl_int clGetEventProfilingInfo(cl_event event,
                                            cl_profiling_info param_name,
                                            size_t param_value_size,
                                            void *param_value,
                                            size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clGetEventProfilingInfo;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clGetEventProfilingInfo");
    return func(event, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Flush and Finish APIs
CL_API_ENTRY cl_int clFlush(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clFlush;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clFlush");
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clFinish(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clFinish;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clFinish");
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Deprecated OpenCL 1.1 APIs
CL_API_ENTRY /* CL_EXT_PREFIX__VERSION_1_1_DEPRECATED */ cl_mem clCreateImage2D(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format *image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void *host_ptr,
    cl_int *errcode_ret) /* CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED */ {
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateImage2D;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateImage2D");
    cl_mem object= func(context, flags, image_format, image_width, image_height,
                   image_row_pitch, host_ptr, errcode_ret);
  #ifdef MEMPROF_SUPPORT
    if(object != NULL) {
      memory_usage::getInstance().updateGPUAlloc((void *)object, getClMemSize(object));
    }
  #endif   
    return object; 
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

// Deprecated OpenCL 2.0 APIs
CL_API_ENTRY /*CL_EXT_PREFIX__VERSION_1_2_DEPRECATED*/ cl_command_queue
clCreateCommandQueue(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties,
                     cl_int *errcode_ret)
/* CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED */ {  // NOLINT
  auto func = deepvan::runtime::OpenCLLibrary::Get()->clCreateCommandQueue;
  if (func != nullptr) {
    LATENCY_LOGGER(3, "clCreateCommandQueue");
    return func(context, device, properties, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

