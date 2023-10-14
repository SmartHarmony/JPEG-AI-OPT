#ifndef DEEPVAN_UTILS_COUNT_DOWN_LATCH_H_
#define DEEPVAN_UTILS_COUNT_DOWN_LATCH_H_

#include <atomic>  // NOLINT(build/c++11)
#include <condition_variable>  // NOLINT(build/c++11)
#include <mutex>  // NOLINT(build/c++11)

#include "deepvan/utils/spinlock.h"

namespace deepvan {
namespace utils {

class CountDownLatch {
 public:
  explicit CountDownLatch(int64_t spin_timeout)
      : spin_timeout_(spin_timeout), count_(0) {}
  CountDownLatch(int64_t spin_timeout, int count)
      : spin_timeout_(spin_timeout), count_(count) {}

  void Wait() {
    if (spin_timeout_ > 0) {
      SpinWaitUntil(count_, 0, spin_timeout_);
    }
    if (count_.load(std::memory_order_acquire) != 0) {
      std::unique_lock<std::mutex> m(mutex_);
      while (count_.load(std::memory_order_acquire) != 0) {
        cond_.wait(m);
      }
    }
  }

  void CountDown() {
    if (count_.fetch_sub(1, std::memory_order_release) == 1) {
      std::unique_lock<std::mutex> m(mutex_);
      cond_.notify_all();
    }
  }

  void Reset(int count) {
    count_.store(count, std::memory_order_release);
  }

  int count() const {
    return count_;
  }

 private:
  int64_t spin_timeout_;
  std::atomic<int> count_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace utils
}  // namespace deepvan

#endif  // DEEPVAN_UTILS_COUNT_DOWN_LATCH_H_

