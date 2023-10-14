//
// Created by Tongping Liu on 3/11/22.
//
#include <dlfcn.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include "./real_apis.h"

#define DEFINE_WRAPPER(name) decltype(::name) * name;
#define INIT_WRAPPER(name, handle) name = (decltype(::name)*)dlsym(handle, #name);

bool realInitialized;

namespace RealX {
    DEFINE_WRAPPER(calloc);
    DEFINE_WRAPPER(sbrk);
    DEFINE_WRAPPER(free);
    DEFINE_WRAPPER(malloc);
    DEFINE_WRAPPER(realloc);
    DEFINE_WRAPPER(memalign);
    DEFINE_WRAPPER(posix_memalign);
    DEFINE_WRAPPER(pthread_create);
    DEFINE_WRAPPER(pthread_join);
    DEFINE_WRAPPER(pthread_exit);
#if 0    
    DEFINE_WRAPPER(clCreateBuffer);
    DEFINE_WRAPPER(clCreateProgramWithSource);
    DEFINE_WRAPPER(clCreateImage);
    DEFINE_WRAPPER(clCreateImage2D);
    DEFINE_WRAPPER(clReleaseMemObject);
    DEFINE_WRAPPER(clGetMemObjectInfo);
    //DEFINE_WRAPPER(clGetProgramObjectInfo);
#endif

    //    cl_mem (*_real_clCreateBuffer) (cl_context, cl_mem_flags, size_t, void *, cl_int *);
    void initializer() {

//		if (realInitialized) return;

        INIT_WRAPPER(calloc, RTLD_NEXT);
        INIT_WRAPPER(sbrk, RTLD_NEXT);
        INIT_WRAPPER(free, RTLD_NEXT);
        INIT_WRAPPER(malloc, RTLD_NEXT);
        INIT_WRAPPER(realloc, RTLD_NEXT);
        INIT_WRAPPER(memalign, RTLD_NEXT);
  	INIT_WRAPPER(posix_memalign, RTLD_NEXT);
    
#if 0 
	// Add customized OpenCL search path here
	// Copied from core/runtime/opencl/opencl_wrapper.cc
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
 
	void * opencl_handle; 
	for (const auto &path : paths) {
	  // VLOG(2) << "Loading OpenCL from " << path;
    	   //opencl_handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    	   opencl_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
           if (opencl_handle != nullptr) {
             break;
           }
        }

/*
  	void *opencl_handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
	      if (opencl_handle == NULL) {
		      fprintf(stderr, "Unable to load libOpenCL.so\n");
		      _exit(2);
	      }
*/

        INIT_WRAPPER(clCreateBuffer, opencl_handle);
        INIT_WRAPPER(clCreateImage, opencl_handle);
        INIT_WRAPPER(clCreateImage2D, opencl_handle);
        INIT_WRAPPER(clCreateProgramWithSource, opencl_handle);
        INIT_WRAPPER(clReleaseMemObject, opencl_handle);
        INIT_WRAPPER(clGetMemObjectInfo, opencl_handle);
#endif
        realInitialized = true;
    }
}

