#ifndef DEEPVAN_CORE_FUTURE_H_
#define DEEPVAN_CORE_FUTURE_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "deepvan/utils/logging.h"
#include "deepvan/export/deepvan.h"

namespace deepvan {
// Wait the call to finish and get the stats if param is not nullptr
struct StatsFuture {
  std::function<void(CallStats *)> wait_fn = [](CallStats *stats) {
    if (stats != nullptr) {
      stats->start_micros = NowMicros();
      stats->end_micros = stats->start_micros;
    }
  };
};

inline void SetFutureDefaultWaitFn(StatsFuture *future) {
  if (future != nullptr) {
    future->wait_fn = [](CallStats * stats) {
      if (stats != nullptr) {
        stats->start_micros = NowMicros();
        stats->end_micros = stats->start_micros;
      }
    };
  }
}

inline void MergeMultipleFutureWaitFn(
    const std::vector<StatsFuture> &org_futures,
    StatsFuture *dst_future) {
  if (dst_future != nullptr) {
    dst_future->wait_fn = [org_futures](CallStats *stats) {
      if (stats != nullptr) {
        stats->start_micros = INT64_MAX;
        stats->end_micros = 0;
        for (auto &org_future : org_futures) {
          CallStats tmp_stats;
          if (org_future.wait_fn != nullptr) {
            org_future.wait_fn(&tmp_stats);
            stats->start_micros = std::min(stats->start_micros,
                                           tmp_stats.start_micros);
            stats->end_micros += tmp_stats.end_micros - tmp_stats.start_micros;
          }
        }
        stats->end_micros += stats->start_micros;
      }
    };
  }
}
}  // namespace deepvan

#endif  // DEEPVAN_CORE_FUTURE_H_
