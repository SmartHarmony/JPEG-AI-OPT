#include "deepvan/core/runtime/cpu/cpu_runtime.h"

#ifdef OPENMP_SUPPORT
#include <omp.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "deepvan/compat/env.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/logging.h"
#include "deepvan/utils/macros.h"

namespace deepvan {
int DeepvanOpenMPThreadCount = 1;

// struct CPUFreq {
//   size_t core_id;
//   float freq;
// };


namespace {

VanState SetOpenMPThreadsAndAffinityCPUs(int omp_num_threads,
                                         SchedulePolicy schedule_policy,
                                         int chunk_size,
                                         const std::vector<size_t> &cpu_ids
                                         ) {
  DeepvanOpenMPThreadCount = omp_num_threads;
  SchedSetAffinity(cpu_ids);
#ifdef OPENMP_SUPPORT
  VLOG(1) << "Set CPU Affinity"
          << ", Threads number: " << omp_num_threads
          << ", Chunk size: " << chunk_size
          << ", Sched Policy: " << schedule_policy
          << ", CPU core IDs: " << MakeString(cpu_ids);

  if (schedule_policy == SCHED_GUIDED) {
    omp_set_schedule(omp_sched_guided, chunk_size);
  } else if (schedule_policy == SCHED_STATIC) {
    omp_set_schedule(omp_sched_static, chunk_size);
  } else if (schedule_policy == SCHED_DYNAMIC) {
    omp_set_schedule(omp_sched_dynamic, chunk_size);
  } else if (schedule_policy == SCHED_AUTO) {
    omp_set_schedule(omp_sched_auto, 0);
  } else {
    LOG(WARNING) << "Unknown schedule policy: " << schedule_policy;
  }

  omp_set_num_threads(omp_num_threads);
#else
  LOG(INFO) << "Set OpenMP Threads and Affinity NO OPENMP SUPPORT";
  UNUSED_VARIABLE(omp_num_threads);
  UNUSED_VARIABLE(schedule_policy);
  UNUSED_VARIABLE(chunk_size);
  LOG(WARNING) << "Set OpenMP threads number failed: OpenMP not enabled.";
#endif

#ifdef OPENMP_SUPPORT
  std::vector<VanState> status(omp_num_threads, VanState::INVALID_ARGS);
#pragma omp parallel for
  for (int i = 0; i < omp_num_threads; ++i) {
    VLOG(1) << "Set affinity for OpenMP thread " << omp_get_thread_num() << "/"
            << omp_get_num_threads();
    status[i] = SchedSetAffinity(cpu_ids);
  }
  for (int i = 0; i < omp_num_threads; ++i) {
    if (status[i] != VanState::SUCCEED)
      return VanState::INVALID_ARGS;
  }
  return VanState::SUCCEED;
#else
  VanState status = SchedSetAffinity(cpu_ids);
  VLOG(1) << "Set affinity without OpenMP: " << MakeString(cpu_ids);
  return status;
#endif
}

} // namespace



VanState CPURuntime::SetAffinityPolicy() {
  int num_threads_hint = cpu_affinity_settings_.num_threads();

  if (num_threads_hint <=0 || num_threads_hint > (int)cpu_count) {
    num_threads_hint = (int)cpu_count;
  }

  CPUAffinityPolicy policy = cpu_affinity_settings_.cpu_affinity_policy();

  if (policy == CPUAffinityPolicy::AFFINITY_NONE) {
#ifdef OPENMP_SUPPORT
    omp_set_num_threads(num_threads_hint);
#else
    LOG(WARNING) << "Set OpenMP threads number failed: OpenMP not enabled.";
#endif
    return VanState::SUCCEED;
  } else if (policy == AFFINITY_CUSTOME_CPUIDS) {
    return SetCPUIDAffinity();
  } else if (policy == AFFINITY_CUSTOME_CLUSTER) {
    return SetClusterAffinity();
  } else if (policy == AFFINITY_EFFICIENT_CORE_FIRST ||
             policy == AFFINITY_PERFORMANCE_CORE_FIRST){
    return SetPerformantOrEfficientCoreFirstAffinity();
  } else if (policy == AFFINITY_BIG_ONLY || 
             policy == AFFINITY_LITTLE_ONLY) {
    return SetBigOrLittleOnlyAffiniy();
  } else if (policy == AFFINITY_POWER_SAVE || 
             policy == AFFINITY_BALANCE || 
             policy == AFFINITY_PERFORMANCE) {
    return SetPerformanceOrBalanceOrPowerSaveAffiniy();
  } else {
    LOG(WARNING) << "Unsupported policy: " << policy;
    return VanState::UNSUPPORTED;
  }
}

VanState CPURuntime::SetPerformanceOrBalanceOrPowerSaveAffiniy(){
  std::vector<size_t> cpu_ids;
  CPUAffinityPolicy policy = cpu_affinity_settings_.cpu_affinity_policy();
  int num_threads_hint = 1;
  int cluster_id = 0;

  if (policy == CPUAffinityPolicy::AFFINITY_PERFORMANCE || 
      policy == CPUAffinityPolicy::AFFINITY_BALANCE) {
    cluster_id = std::max(0, (int)clusters.size() - 2);
  }

  cpu_ids = clusters[cluster_id].cpu_ids;

  if (policy == CPUAffinityPolicy::AFFINITY_PERFORMANCE || 
      policy == CPUAffinityPolicy::AFFINITY_POWER_SAVE) {
    size_t size = cpu_ids.size()/2;
    if (size > 0) {
      num_threads_hint = num_threads_hint << 1;
    }
  }

  return SetOpenMPThreadsAndAffinityCPUs(num_threads_hint,
                                         SCHED_STATIC,
                                         0, cpu_ids); 
}

VanState CPURuntime::SetBigOrLittleOnlyAffiniy() {
  std::vector<size_t> cpu_ids;
  CPUAffinityPolicy policy = cpu_affinity_settings_.cpu_affinity_policy();
  int num_threads_hint = cpu_affinity_settings_.num_threads();

  if (num_threads_hint <= 0 || num_threads_hint >= (int)cpu_count) {
    num_threads_hint = cpu_count;
  }

  if (policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
    cpu_ids = clusters.back().cpu_ids;
    for (int i = clusters.size() - 2; i > 0; i--)
      cpu_ids.insert(cpu_ids.end(), clusters[i].cpu_ids.begin(), clusters[i].cpu_ids.end());
  } else if (policy == CPUAffinityPolicy::AFFINITY_LITTLE_ONLY) {
    cpu_ids = clusters.front().cpu_ids;
  } else {
    LOG(WARNING) << "Unsupported affinity policy in BigOrLittleAffinity";
    return VanState::UNSUPPORTED;
  }

  num_threads_hint = std::max(num_threads_hint, 0);
  num_threads_hint = std::min(num_threads_hint, (int)cpu_ids.size());

  return SetOpenMPThreadsAndAffinityCPUs(num_threads_hint,
                                         SCHED_STATIC,
                                         0, cpu_ids); 

}

VanState CPURuntime::SetPerformantOrEfficientCoreFirstAffinity() {
  std::vector<size_t> cpu_ids;
  CPUAffinityPolicy policy = cpu_affinity_settings_.cpu_affinity_policy();
  int num_threads_hint = cpu_affinity_settings_.num_threads();

  if (num_threads_hint <= 0 || num_threads_hint >= (int)cpu_count) {
    num_threads_hint = cpu_count;
  }

  if (policy == CPUAffinityPolicy::AFFINITY_EFFICIENT_CORE_FIRST) {
    for (auto c: clusters) {
      cpu_ids.insert(cpu_ids.end(), c.cpu_ids.begin(), c.cpu_ids.end());
    }
  } else if (policy == CPUAffinityPolicy::AFFINITY_PERFORMANCE_CORE_FIRST) {
    for (auto it = clusters.rbegin(); it != clusters.rend(); it++) {
      cpu_ids.insert(cpu_ids.end(), it->cpu_ids.begin(), it->cpu_ids.end());
    }
  } else {
    LOG(WARNING) << "Unsupported affinity policy in PerformantOrEfficientAffinity";
    return VanState::UNSUPPORTED;
  }

  cpu_ids.resize(num_threads_hint);

  SchedulePolicy omp_sched_policy = SCHED_STATIC;
  int chunk_size = 0;
  if (UserMultipleClusters(cpu_ids)) {
    omp_sched_policy = SCHED_GUIDED;
    chunk_size = 1;
  }

  return SetOpenMPThreadsAndAffinityCPUs(num_threads_hint,
                                         omp_sched_policy,
                                         chunk_size,
                                         cpu_ids); 

}

VanState CPURuntime::SetClusterAffinity() {
  std::vector<size_t> cpu_ids;
  std::vector<size_t> cluster_ids = cpu_affinity_settings_.ids();
  SchedulePolicy omp_sched_policy = cpu_affinity_settings_.omp_schedule_policy();
  int num_threads_hint = cpu_affinity_settings_.num_threads();
  int chunk_size = cpu_affinity_settings_.omp_chunk_size();

  for (auto id: cluster_ids) {
    if (id >= clusters.size()) {
      LOG(ERROR) << "Invalid cluster id: " << id;
    } else {
      cpu_ids.insert(cpu_ids.end(), clusters[id].cpu_ids.begin(), clusters[id].cpu_ids.end());
    }
  }

  if (cpu_ids.empty()) {
      LOG(ERROR) << "No valid cpu ids assigned, the little core will be used.";
      cpu_ids = clusters[0].cpu_ids;
  }

  num_threads_hint = std::max(num_threads_hint, 0);
  num_threads_hint = std::min(num_threads_hint, (int)cpu_ids.size());

  if (omp_sched_policy == SCHED_DEFAULT) {
    omp_sched_policy = clusters.size() > 1 ? SCHED_GUIDED: SCHED_STATIC;
  }

  if (chunk_size < 0) {
    chunk_size = clusters.size() > 1 ? 1: 0;
  }

  return SetOpenMPThreadsAndAffinityCPUs(num_threads_hint,
                                         omp_sched_policy,
                                         chunk_size,
                                         cpu_ids);
}

VanState CPURuntime::SetCPUIDAffinity() {
  int chunk_size = cpu_affinity_settings_.omp_chunk_size(); 
  SchedulePolicy omp_sched_policy = cpu_affinity_settings_.omp_schedule_policy();

  bool use_multiple_clusters = false;

  if (chunk_size < 0 || omp_sched_policy == SCHED_DEFAULT) {
    use_multiple_clusters = UserMultipleClusters(cpu_affinity_settings_.ids());
  }

  if (chunk_size < 0) {
    chunk_size = use_multiple_clusters ? 1: 0;
  }

  if (omp_sched_policy == SCHED_DEFAULT) {
    omp_sched_policy = SCHED_STATIC;
    omp_sched_policy = use_multiple_clusters ? SCHED_GUIDED: SCHED_STATIC;
  }

  return SetOpenMPThreadsAndAffinityCPUs(cpu_affinity_settings_.num_threads(),
                                         omp_sched_policy, chunk_size, 
                                         cpu_affinity_settings_.ids());
}

bool CPURuntime::UserMultipleClusters(std::vector<size_t> &cpu_ids) {
  uint64_t freq = cpu_max_freqs[cpu_ids[0]];
  for (auto id: cpu_ids) {
    if (freq != cpu_max_freqs[id]) return true;
    freq = cpu_max_freqs[id];
  }
  return false;
}

} // namespace deepvan
