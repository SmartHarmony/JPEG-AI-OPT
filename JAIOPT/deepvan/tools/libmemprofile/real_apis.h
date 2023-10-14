//
// Created by Tongping Liu on 3/11/22.
//

#ifndef COCOPIEAPIEXAMPLE_REAL_H
#define COCOPIEAPIEXAMPLE_REAL_H

#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <malloc.h>
#include <CL/opencl.h>

#define DECLARE_WRAPPER(name) extern decltype(::name) * name;

#define WRAP(x) _real_##x

extern bool realInitialized;

namespace RealX {
    void initializer();
    DECLARE_WRAPPER(free);
    DECLARE_WRAPPER(calloc);
    DECLARE_WRAPPER(malloc);
    DECLARE_WRAPPER(realloc);
    DECLARE_WRAPPER(memalign);
    DECLARE_WRAPPER(posix_memalign);
#if 0
    DECLARE_WRAPPER(clCreateBuffer);
    DECLARE_WRAPPER(clCreateImage);
    DECLARE_WRAPPER(clCreateImage2D);
    DECLARE_WRAPPER(clCreateProgramWithSource);
    DECLARE_WRAPPER(clReleaseMemObject);
    DECLARE_WRAPPER(clGetMemObjectInfo);
#endif
};

#endif //COCOPIEAPIEXAMPLE_REAL_H

