#ifndef _OBJECTS_MAP_
#define _OBJECTS_MAP_

#include <pthread.h>
#include <unordered_map>

#define PC_COUNT_BACKTRACE 2
#define SKIP_COUNT_BACKTRACE 3
/*
template < class Key,         /cxx-stl/gnu-libstdc++/4.9/include/bits/functional_hash.h:58:12: note: template is declared here
    struct hash;
                           // unordered_map::key_type
           class T,                                      // unordered_map::mapped_type
           class Hash = hash<Key>,                       // unordered_map::hasher
           class Pred = equal_to<Key>,                   // unordered_map::key_equal
           class Alloc = allocator< pair<const Key,T> >  // unordered_map::allocator_type
           > class unordered_map;
*/
/* 
   We will use two maps for both CPU/GPU allocations here, as there is no difference for detecting purpose. 
     One map is called _activeMap that holds the callsite of every active object. 
  
     The other map is called _csMap that holds the allocation information of every callsite, which is used to identify potential memory leaks. 

     Upon each allocation, we will get the callsite, and then add the current object into _activeMap, with the key of object address.
     Upon each deallocation, we will check the _activeMap to obtain its allocation callsite, remove it from _activeMap. Then we will use the 
     callsite to check _csMap, and then decrement the allocation count there. 

   Note: as we will insert/free entries upon allocations and deallocations, which will invoke some additional allocations/deallocations, 
     the operations may create an endless loop. One alternative is to use a perthread variable to store the status of the thread, avoiding 
     the loop for all malloc functions, which is a little ugly. Another way is to use an internal heap for hash map, so that we won't 
     have this issue. We will try the second approach here. 

   Note: currently, stl's unordered_map is suitable for the hash map. However, it may introduce high contention, as it is using the same lock 
     for all entries. One alternative is to use a custom hashmap, as https://github.com/UTSASRG/Guarder/blob/master/hashmap.hh. 
     But let's use the stl one at first. 
 */ 

extern "C" { 
  extern thread_local bool _insideMM;
  void get_backtrace(unsigned long * pc); 
};

class Callsite {
  public:
    unsigned long callsite[PC_COUNT_BACKTRACE];

    bool operator()(const Callsite& cs) const {
     return (callsite[0] == cs.callsite[0] && callsite[1] == cs.callsite[1]);
    }  
 };

class hashCallsite {
  public:
   size_t operator()(const Callsite& p) const {
     return p.callsite[0] + p.callsite[1];
   }

};

class csEqual {
  public:
   bool operator()(const Callsite & p1, const Callsite & p2) const {
     return (p1.callsite[0] == p2.callsite[0]) && (p1.callsite[1] == p2.callsite[1]); 
   }

};

class ObjectsMap {
  public:
  ObjectsMap() {
    pthread_mutex_init(&_lock, NULL);
  }


  // Insert one item into the objs map.
  void onAllocation(void * ptr, Callsite & cs) {
   
    //fprintf(stderr, "put ptr %p\n", ptr); 
    lock();
   // We will always insert the object into the _activeMap, as there 
   // are no two objects with the same address/pointer    
    _activeMap.insert(std::make_pair(ptr, cs)); 

    // Check whether the entry exists in the callsite map. 
    // If it exists, increment the allocation number.
    csIterator csi = _csMap.find(cs); 
    if(csi != _csMap.end()) {
        csi->second++; 
    } 
    else {
      int alloc = 1; 
      _csMap.insert(std::make_pair(cs, alloc)); 
    }
    
    unlock(); 

     
  }

  void onFree(void * ptr) {
    size_t size = 0; 

    objectIterator entry;
    entry = _activeMap.find(ptr);

    if (entry != _activeMap.end()) {
      lock(); 

      // Get the item from the hash map. 
      csIterator cs = _csMap.find(entry->second); 
      if(cs != _csMap.end()) {
        cs->second--; 
      }
      
      _activeMap.erase(entry);

      unlock();
    }
  }

  private:
  using objectKey = void *;
  using objectValue = Callsite;
  using objectEntry = std::pair<objectKey, objectValue>;
  std::unordered_map<objectKey, objectValue, std::hash<objectKey> > _activeMap;
  using objectIterator = std::unordered_map<objectKey, objectValue>::iterator;


  using csKey = Callsite; 
  using csValue = unsigned long; 
  using csEntry = std::pair<csKey, csValue>; 
  std::unordered_map<csKey, csValue, hashCallsite, csEqual> _csMap;
  using csIterator = std::unordered_map<csKey, csValue, hashCallsite, csEqual>::iterator;

  
  //  std::unordered_map<key, value, std::hash<key>, std::equal_to<key>> _activeMap;


  pthread_mutex_t _lock;

  void lock() {
    pthread_mutex_lock(&_lock);
  }

  void unlock() {
    pthread_mutex_unlock(&_lock);
  }
}; 

#endif
