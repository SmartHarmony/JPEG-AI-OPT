#ifndef DEEPVAN_UTILS_RWLOCK_H_
#define DEEPVAN_UTILS_RWLOCK_H_

#include <condition_variable>  // NOLINT(build/c++11)
#include <mutex>  // NOLINT(build/c++11)

#include "deepvan/utils/logging.h"
#include "deepvan/utils/macros.h"

namespace deepvan {
namespace utils {

class RWMutex {
 public:
  RWMutex() : counter_(0), waiting_readers_(0), waiting_writers_(0) {}
  ~RWMutex() = default;

  int counter_;  // -1 for writer, 0 for nobody, 1~n for reader
  int waiting_readers_;
  int waiting_writers_;
  std::mutex mutex_;
  std::condition_variable reader_cv_;
  std::condition_variable writer_cv_;

  DISABLE_COPY_AND_ASSIGN(RWMutex);
};

// Writer first
class ReadLock {
 public:
  explicit ReadLock(RWMutex *rw_mutex): rw_mutex_(rw_mutex) {
    CONDITIONS_NOTNULL(rw_mutex);
    std::unique_lock<std::mutex> lock(rw_mutex->mutex_);
    rw_mutex->waiting_readers_ += 1;
    rw_mutex->reader_cv_.wait(lock, [&]() -> bool {
      return rw_mutex->waiting_writers_ == 0 && rw_mutex->counter_ >= 0;
    });
    rw_mutex->waiting_readers_ -= 1;
    rw_mutex->counter_ += 1;
  }
  ~ReadLock() {
    std::unique_lock<std::mutex> lock(rw_mutex_->mutex_);
    rw_mutex_->counter_ -= 1;
    if (rw_mutex_->waiting_writers_ > 0) {
      if (rw_mutex_->counter_ == 0) {
        rw_mutex_->writer_cv_.notify_one();
      }
    }
  }

 private:
  RWMutex *rw_mutex_;

  DISABLE_COPY_AND_ASSIGN(ReadLock);
};

class WriteLock {
 public:
  explicit WriteLock(RWMutex *rw_mutex): rw_mutex_(rw_mutex) {
    CONDITIONS_NOTNULL(rw_mutex);
    std::unique_lock<std::mutex> lock(rw_mutex->mutex_);
    rw_mutex->waiting_writers_ += 1;
    rw_mutex->writer_cv_.wait(lock, [&]() -> bool {
      return rw_mutex->counter_ == 0;
    });
    rw_mutex->waiting_writers_ -= 1;
    rw_mutex->counter_ -= 1;
  }
  ~WriteLock() {
    std::unique_lock<std::mutex> lock(rw_mutex_->mutex_);
    rw_mutex_->counter_ = 0;
    if (rw_mutex_->waiting_writers_ > 0) {
      rw_mutex_->writer_cv_.notify_one();
    } else {
      rw_mutex_->reader_cv_.notify_all();
    }
  }

 private:
  RWMutex *rw_mutex_;

  DISABLE_COPY_AND_ASSIGN(WriteLock);
};

}  // namespace utils
}  // namespace deepvan

#endif  // DEEPVAN_UTILS_RWLOCK_H_
