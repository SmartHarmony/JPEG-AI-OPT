// Created by Tongping Liu on 3/11/22.
#include <dlfcn.h>
#include <stdio.h>
#include <string>
#include <dlfcn.h>
#include <CL/opencl.h>

//#include <unwind.h>
#include "libunwind.h"

#include "real_apis.h"
#include "memory_usage.h"

char localBuffer[131072];
char * localPtr = NULL;
char * localPtrEnd;

typedef enum {
    E_INIT_NOT = 0,
    E_INIT_WORKING,
    E_INIT_DONE
} eInitStatus;
eInitStatus _initStatus = E_INIT_NOT;


void initializer(void) {
  if(_initStatus == E_INIT_NOT) {
      _initStatus = E_INIT_WORKING;
      localPtr = localBuffer;
      localPtrEnd = &localBuffer[131072];
      RealX::initializer();
      memory_usage::getInstance().initialize();
  }
  _initStatus = E_INIT_DONE;
}


thread_local bool _insideMM; 

//__attribute__((destructor)) void finalizer() {
//  memory_usage::getInstance().printCPUMemoryUsage();
//  memory_usage::getInstance().printGPUMemoryUsage();
//}

extern "C" {

  void get_backtrace(unsigned long * pc) {
    unw_cursor_t    cursor;
    unw_context_t   context;

    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    int i = 0; 
    int skip = 0; 
    while (unw_step(&cursor) > 0)
    {
	skip++; 
	if(skip < SKIP_COUNT_BACKTRACE) {
	  continue;
	}

	if(i < PC_COUNT_BACKTRACE) {	
	  // Now we will store the pc in the given array
          unw_get_reg(&cursor, UNW_REG_IP, &pc[i++]);
	}
    }
    return; 
  }

	
  // Intercept malloc() api
  void * malloc(size_t sz) {
    void * ptr = NULL;

    if(_initStatus != E_INIT_DONE) {
      if(_initStatus == E_INIT_NOT ) {
          initializer();
      }
      ptr = localPtr;
      localPtr += sz;
      return ptr;
    }

    ptr = RealX::malloc(sz);
    // Update the cpu allocation when the allocation is successful
    if(ptr != NULL && _insideMM == false) {
      _insideMM = true; 
      memory_usage::getInstance().updateCPUAlloc(ptr, malloc_usable_size(ptr));
      _insideMM = false; 
    }
    return ptr;
  }

  // Intercept calloc() api
  void * calloc(size_t nelem, size_t elsize) {
    void * ptr = NULL;
   
    if(_initStatus == false) {
      ptr = localPtr; 
      localPtr += nelem * elsize;
      return ptr;
    }

    ptr = RealX::calloc(nelem, elsize);
    if( ptr != NULL && _insideMM == false) {
      _insideMM = true; 
      memory_usage::getInstance().updateCPUAlloc(ptr, malloc_usable_size(ptr));
      _insideMM = false; 
    }

    return ptr;
  }

  // Intercept realloc() api
  void * realloc(void * ptr, size_t sz) {
   
    size_t oldsz = 0; 
    if(ptr != NULL) {
      oldsz = malloc_usable_size(ptr);
    }

    void * newptr = RealX::realloc(ptr, sz);
   
    // As realloc may increase/dealloc the size of the object
    // we should handle it differently
    if(ptr == NULL && _insideMM == false) {
      _insideMM = true;
      memory_usage::getInstance().updateCPUAlloc(newptr, malloc_usable_size(newptr));
      _insideMM = false;
    }
    else {
      // The original object is freed now. 
      memory_usage::getInstance().updateCPUFree(ptr, oldsz);

      if(sz != 0 && newptr != NULL && _insideMM == false) {
        _insideMM = true;
        // The new object is allocated now.  
        memory_usage::getInstance().updateCPUAlloc(newptr, malloc_usable_size(newptr));
        _insideMM = false;
      }
    } 
       
    return newptr;
  }
 
  // Handle the free() api 
  // Note that for CPU memory allocations, there is no need to add the objects into 
  // the hashmap, as we could always rely on malloc_usable_size to obtain the size of 
  // an object, so that we could deduct the size of an object correctly.
  // However, this mechanism may not work for some random allocator 
  void free(void * ptr) {
    if(ptr == nullptr) 
      return;

    // Note that if the pointer is located on the temporary buffer, then
    // we don't need to return to the normal allocator. Otherwise, 
    // the program will crash
    if(ptr >= &localBuffer && ptr < localPtrEnd ) {
        return;
    }

    //TODO : we may need to utilize other mechanisms to get the size of the corresponding object in a non-linux machine. 
    size_t sz = malloc_usable_size(ptr); 

    RealX::free(ptr);

    // Decrease the size of theobject
    memory_usage::getInstance().updateCPUFree(ptr, sz);
  }

  // Handle the posix_memalign() 
  int posix_memalign(void **memptr, size_t alignment, size_t size) {
    int ret = RealX::posix_memalign(memptr, alignment, size);

    // Only update the cpu allocation information when the allocation is successful
    if(ret == 0 && _insideMM == false) {
      _insideMM = true;
      int newsize = malloc_usable_size(*memptr);
      memory_usage::getInstance().updateCPUAlloc(*memptr, newsize);
      _insideMM = false;
    }
    return ret;
  }

  // Handle memalign() api, which is the one mainly used by deepvan
  void * memalign(size_t alignment, size_t size) {
    void * ptr = NULL; 

    // Invoke the real memalign of the standard library
    ptr = RealX::memalign(alignment, size);

    // Only update the cpu allocation information when the allocation is successful
    if(ptr != NULL && _insideMM == false) {
      _insideMM = true;
      // Due to the alignment, we have to adjust the size based on alignment too
      // as the newsize is the real size of the allocation
      int newsize = malloc_usable_size(ptr);
      memory_usage::getInstance().getInstance().updateCPUAlloc(ptr, newsize);
      _insideMM = false;
    }

    return ptr;
  } 

#if 0
  // The following functions has been moved to core/runtime/opencl/opencl_wrapper.cc
  // as we can't have two places that intercept the same function. 
  // That is, there is no need for these functions if we are combining with deepvan's code base. 

  inline size_t getClMemSize(cl_mem memobj) {
    size_t size = 0; 
    
    if(RealX::clGetMemObjectInfo(memobj, CL_MEM_SIZE, sizeof(size), &size, NULL) == CL_SUCCESS) {
       return size;
    }
    else {
    	return 0;
    }
  }
  // Intercept OpenCL's clCreateBuffer or new cl::Buffer(). 
  cl_mem clCreateBuffer (
    cl_context context, 
  	cl_mem_flags flags,
  	size_t size,
  	void *host_ptr,
  	cl_int *errcode_ret) {
    cl_mem object; 
    
    if(_initStatus != E_INIT_DONE) {
      if(_initStatus == E_INIT_NOT ) {
          initializer();
      }
    }

    object = RealX::clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
    if(errcode_ret == CL_SUCCESS) {
    //	memory_usage::getInstance().updateGPUAlloc(getClMemSize(object));
    }
    return object;
  }

  // Intercept OpenCL's clCreateImage or cl::Image2D
  cl_mem clCreateImage(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret) {
    cl_mem object; 

    if(_initStatus != E_INIT_DONE) {
      if(_initStatus == E_INIT_NOT ) {
          initializer();
      }
    }
    object = RealX::clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
    if(object != NULL && errcode_ret == CL_SUCCESS) {
    	memory_usage::getInstance().updateGPUAlloc(getClMemSize(object));
    }

    return object;
  }

  cl_mem clCreateImage2D ( 	
	cl_context context,
  	cl_mem_flags flags,
  	const cl_image_format *image_format,
  	size_t image_width,
  	size_t image_height,
  	size_t image_row_pitch,
  	void *host_ptr,
  	cl_int *errcode_ret) {
    cl_mem object; 

    if(_initStatus != E_INIT_DONE) {
      if(_initStatus == E_INIT_NOT ) {
          initializer();
      }
    }
    object = RealX::clCreateImage2D(context, flags, image_format, image_width, image_height, image_row_pitch, host_ptr, errcode_ret);
    if(object != NULL && errcode_ret == CL_SUCCESS) {
    	memory_usage::getInstance().updateGPUAlloc(getClMemSize(object));
    }

    return object;

  }

  // Handle OpenCL's clCreateProgramWithSource. Again, this is the only function that has been invoked so far
  // TODO: extend this if deepvan has been changed in the future
  cl_program clCreateProgramWithSource(
        cl_context context,
        cl_uint count,
        const char **strings,
        const size_t *lengths,
        cl_int *errcode_ret) {
  cl_program object;

    if(_initStatus != E_INIT_DONE) {
      if(_initStatus == E_INIT_NOT ) {
          initializer();
      }
    }
  object = RealX::clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);

  if(errcode_ret == CL_SUCCESS) {
    size_t deviceCount; 
    size_t sizeArray[8]; // No need to use malloc here, as the number of device is typically very small 
    clGetProgramInfo(object, CL_PROGRAM_NUM_DEVICES, sizeof(deviceCount), &deviceCount, NULL);

    clGetProgramInfo(object, CL_PROGRAM_BINARY_SIZES, sizeof(sizeArray), sizeArray, NULL); 
    size_t size = 0; 

    for(size_t i = 0; i < deviceCount; i++) {
      size += sizeArray[i];
    }

    memory_usage::getInstance().updateGPUAlloc(size);
  }

  return object;
}

// Handle the release of memory object.  
cl_int clReleaseMemObject (cl_mem memobj) {
    cl_int error; 
  
    // Note: we will have to request to update the free information
    // before clReleaseMemObject, otherwise, there is no way to obtain
    // the object's size information and the program will crash
    if(memobj != NULL) {  
    	memory_usage::getInstance().updateGPUAlloc(getClMemSize(memobj));
    }

    error = RealX::clReleaseMemObject(memobj);
    return error;
  }
#endif

}; 
