#ifndef DEEPVAN_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
#define DEEPVAN_CORE_RUNTIME_CPU_CPU_RUNTIME_H_

#include <cassert>
#include <memory>
#include <vector>

#include "deepvan/export/deepvan.h"
#include "deepvan/utils/logging.h"
#include "deepvan/utils/macros.h"

namespace deepvan {
extern int DeepvanOpenMPThreadCount;

class CPUAffinityPolicySettings {
  public:
    CPUAffinityPolicySettings(int num_threads, CPUAffinityPolicy policy):
      num_threads_(num_threads), cpu_affinity_policy_(policy), omp_chunk_size_(0),
      omp_sched_policy_(SCHED_STATIC), enable_autoset_(false) {
        ids_.clear();
      }

    CPUAffinityPolicySettings():
      num_threads_(2),
      cpu_affinity_policy_(CPUAffinityPolicy::AFFINITY_PERFORMANCE),
      omp_chunk_size_(0),
      omp_sched_policy_(SCHED_STATIC),
      enable_autoset_(true) {
        ids_.clear();
      }

    CPUAffinityPolicySettings(int num_threads, CPUAffinityPolicy cpu_affinity_policy,
                             int omp_chunk_size, SchedulePolicy omp_sched_policy,
                             std::vector<size_t> &ids, bool enable_autoset) {
      num_threads_ = num_threads;
      cpu_affinity_policy_ = cpu_affinity_policy;
      omp_chunk_size_ = omp_chunk_size;
      omp_sched_policy_ = omp_sched_policy;
      ids_ = ids;
      enable_autoset_ = enable_autoset;                        
    }

    CPUAffinityPolicySettings(const CPUAffinityPolicySettings &settings) {
      num_threads_         = settings.num_threads_;
      cpu_affinity_policy_ = settings.cpu_affinity_policy_;
      omp_chunk_size_      = settings.omp_chunk_size_;
      omp_sched_policy_    = settings.omp_sched_policy_;
      ids_                 = settings.ids_;
      enable_autoset_      = settings.enable_autoset_;   
    }

    void reset(int num_threads, 
               CPUAffinityPolicy cpu_affinity, 
               int omp_chunk_size, 
               SchedulePolicy omp_sched_policy, 
               const std::vector<size_t> &ids,
               bool enable_autoset = true) {
      num_threads_ = num_threads;
      cpu_affinity_policy_ = cpu_affinity;
      omp_chunk_size_ = omp_chunk_size;
      omp_sched_policy_ = omp_sched_policy;
      ids_ = ids;
      enable_autoset_ = enable_autoset;
    }

    inline bool enable_autoset() const { return enable_autoset_; }
    inline int num_threads() const { return num_threads_; }
    inline int omp_chunk_size() const { return omp_chunk_size_; }
    inline SchedulePolicy omp_schedule_policy() const { return omp_sched_policy_; }
    inline CPUAffinityPolicy cpu_affinity_policy() const { return cpu_affinity_policy_; }
    inline std::vector<size_t>& ids() {return ids_; }

  private:
    bool enable_autoset_;
    int num_threads_;
    int omp_chunk_size_;
    SchedulePolicy omp_sched_policy_;
    CPUAffinityPolicy cpu_affinity_policy_;
    std::vector<size_t> ids_; // a unique data structure holding either cluster ids or cpu ids
};


class CPURuntime {
public:
  CPURuntime(const CPUAffinityPolicySettings &cpu_affinity_settings)
      : cpu_affinity_settings_(cpu_affinity_settings), gemm_context_(nullptr) {

        cpu_count = GetCPUCount();
        CONDITIONS(cpu_count > 0);

        VanState status = GetCPUClusters(clusters);
        CONDITIONS(status == VanState::SUCCEED);

        status = GetCPUMaxFreq(&cpu_max_freqs);
        CONDITIONS(status == VanState::SUCCEED);

        status = SetAffinityPolicy();
  }

  ~CPURuntime() = default;

  CPUAffinityPolicySettings& cpu_affinity_settings() {return cpu_affinity_settings_; }

  VanState SetAffinityPolicy();

private:
  VanState SetPerformanceOrBalanceOrPowerSaveAffiniy();
  VanState SetBigOrLittleOnlyAffiniy();
  VanState SetPerformantOrEfficientCoreFirstAffinity();
  VanState SetClusterAffinity();
  VanState SetCPUIDAffinity();
  bool UserMultipleClusters(std::vector<size_t> &cpu_ids);

  CPUAffinityPolicySettings cpu_affinity_settings_;
  std::vector<CPUCluster> clusters;
  std::vector<uint64_t> cpu_max_freqs;
  size_t cpu_count;
  void *gemm_context_;
};
} // namespace deepvan

#endif // DEEPVAN_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
