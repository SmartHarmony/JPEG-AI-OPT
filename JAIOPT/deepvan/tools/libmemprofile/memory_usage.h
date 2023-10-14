#ifndef __MEMORY_USAGE_H__
#define __MEMORY_USAGE_H__

#include "objects_map.h"

class memory_usage {

  public:
  memory_usage() { }

  void initialize() {
    // Initialize all counters to be 0
    _peakUsage_GPU = 0; 
    _allocs_GPU = 0; 
    _frees_GPU = 0; 
    _currentUsage_GPU = 0;
    _peakUsage_CPU = 0; 
    _allocs_CPU = 0; 
    _frees_CPU = 0; 
    _currentUsage_CPU = 0;
  }

  static memory_usage & getInstance() {
    static char buf[sizeof(memory_usage)];
    static memory_usage * theOneObject = new (buf) memory_usage();
    return *theOneObject;
  }

  void onAllocation(void *ptr) {
    // Obtain the callsite. 
    Callsite cs; 
    
    get_backtrace((unsigned long *)&cs); 

    // Update ObjectsMap upon the allocation	  
    _objMap.onAllocation(ptr, cs); 
  }

  void onFree(void * ptr) {
    // Update ObjectsMap upon the free 
    _objMap.onFree(ptr); 
  }


  void updateGPUAlloc(void * ptr, size_t size) {
    // If the current thread is inside MM, then possibly we are allocating 
    // an object for the hash map. Then there is no need to record the size and and record the 
    // callsite and others, as we only focus on application/XGen's code. 
    if(!_insideMM) {
	_insideMM = true; 

    	// Add the ptr and size as a pair to the hash map.
    	_allocs_GPU ++;
    	_currentUsage_GPU += size; 
    	if(_currentUsage_GPU > _peakUsage_GPU) {
      	  _peakUsage_GPU = _currentUsage_GPU;
    	}

        // Add the current object into the objects' map. 
        onAllocation(ptr); 

	_insideMM = false; 
    }
  } 

  void updateGPUFree(void * ptr, size_t size) {
    if(!_insideMM) {
      _insideMM = true; 

      // Update the counters
      _frees_GPU++; 
      _currentUsage_GPU -= size;

      // Remove the object from the objects' map
      onFree(ptr); 

      _insideMM = false; 
    }
  }

  void updateCPUAlloc(void * ptr, size_t size) {
    if(!_insideMM) {
	_insideMM = true; 

       _allocs_CPU ++;
       _currentUsage_CPU += size; 
       if(_currentUsage_CPU > _peakUsage_CPU) {
         _peakUsage_CPU = _currentUsage_CPU;
       }
        
       // Add the current object into the objects' map. 
       onAllocation(ptr); 
       _insideMM = false; 
    }
  }

  void updateCPUFree(void * ptr, size_t size) {
    if(!_insideMM) {
	_insideMM = true; 

        _frees_CPU++; 
        _currentUsage_CPU -= size;
      
	// Remove the object from the objects' map
        onFree(ptr); 
	
        _insideMM = false; 
    }
  }

  void obtainCPUMemoryUsage(size_t * peak, size_t * current) {
    *peak = _peakUsage_CPU;
    *current = _currentUsage_CPU;
  }
  
  void obtainGPUMemoryUsage(size_t * peak, size_t * current) {
    *peak = _peakUsage_GPU;
    *current = _currentUsage_GPU;
  }
  
  ssize_t getCPUMemoryUsage() {
    return (ssize_t)_peakUsage_CPU;
  }
  
  ssize_t getGPUMemoryUsage() {
    return (ssize_t) _peakUsage_GPU;
  }


  void printCPUMemoryUsage(void) {
    fprintf(stderr, "CPU memory usage information:\n");
    fprintf(stderr, "\t allocs: %llu, peak usage %llu\n", _allocs_CPU, _peakUsage_CPU);
    fprintf(stderr, "\t frees: %llu, current usage %llu\n", _frees_CPU, _currentUsage_CPU);
  }

  void printGPUMemoryUsage(void) {
    fprintf(stderr, "GPU memory usage information:\n");
    fprintf(stderr, "\t allocs: %llu, peak usage %llu\n", _allocs_GPU, _peakUsage_GPU);
    fprintf(stderr, "\t frees: %llu, current usage %llu\n", _frees_GPU, _currentUsage_GPU);
  }


  private:
   unsigned long long _peakUsage_CPU; 
   unsigned long long _allocs_CPU; 
   unsigned long long _frees_CPU;
   unsigned long long _currentUsage_CPU;
    
   unsigned long long _peakUsage_GPU; 
   unsigned long long _allocs_GPU; 
   unsigned long long _frees_GPU;
   unsigned long long _currentUsage_GPU;
   ObjectsMap _objMap; 
};


#endif
