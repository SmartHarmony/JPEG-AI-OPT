#ifndef DEEPVAN_COMPAT_POSIX_TIME_H_
#define DEEPVAN_COMPAT_POSIX_TIME_H_

#include <sys/time.h>

#include <cstddef>

namespace deepvan {
namespace compat {
namespace posix {

inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

}  // namespace posix
}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_POSIX_TIME_H_
