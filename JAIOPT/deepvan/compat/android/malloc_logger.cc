#include "deepvan/compat/android/malloc_logger.h"

#include <malloc.h>

#include <string>
#include <utility>

namespace deepvan {
namespace compat {

namespace {
struct mallinfo LogMallinfoChange(std::ostringstream *oss,
                                  const std::string &name,
                                  const struct mallinfo curr,
                                  const struct mallinfo prev) {
  if (prev.arena != curr.arena) {
    (*oss) << "[" << name << "] "
           << "Non-mmapped space allocated (bytes): " << curr.arena
           << ", diff: " << ((int64_t)curr.arena - (int64_t)prev.arena);
  }
  if (prev.ordblks != curr.ordblks) {
    (*oss) << "[" << name << "] "
           << "Number of free chunks: " << curr.ordblks << ", diff: "
           << ((int64_t)curr.ordblks - (int64_t)prev.ordblks);
  }
  if (prev.smblks != curr.smblks) {
    (*oss) << "[" << name << "] "
           << "Number of free fastbin blocks: " << curr.smblks
           << ", diff: " << ((int64_t)curr.smblks - (int64_t)prev.smblks);
  }
  if (prev.hblks != curr.hblks) {
    (*oss) << "[" << name << "] "
           << "Number of mmapped regions: " << curr.hblks
           << ", diff: " << ((int64_t)curr.hblks - (int64_t)prev.hblks);
  }
  if (prev.hblkhd != curr.hblkhd) {
    (*oss) << "[" << name << "] "
           << "Space allocated in mmapped regions (bytes): " << curr.hblkhd
           << ", diff: " << ((int64_t)curr.hblkhd - (int64_t)prev.hblkhd);
  }
  if (prev.usmblks != curr.usmblks) {
    (*oss) << "[" << name << "] "
           << "Maximum total allocated space (bytes): " << curr.usmblks
           << ", diff: "
           << ((int64_t)curr.usmblks - (int64_t)prev.usmblks);
  }
  if (prev.fsmblks != curr.fsmblks) {
    (*oss) << "[" << name << "] "
           << "Space in freed fastbin blocks (bytes): " << curr.fsmblks
           << ", diff: "
           << ((int64_t)curr.fsmblks - (int64_t)prev.fsmblks);
  }
  if (prev.uordblks != curr.uordblks) {
    (*oss) << "[" << name << "] "
           << "Total allocated space (bytes): " << curr.uordblks
           << ", diff: "
           << ((int64_t)curr.uordblks - (int64_t)prev.uordblks);
  }
  if (prev.fordblks != curr.fordblks) {
    (*oss) << "[" << name << "] "
           << "Total free space (bytes): " << curr.fordblks << ", diff: "
           << ((int64_t)curr.fordblks - (int64_t)prev.fordblks);
  }
  if (prev.keepcost != curr.keepcost) {
    (*oss) << "[" << name << "] "
           << "Top-most, releasable space (bytes): " << curr.keepcost
           << ", diff: "
           << ((int64_t)curr.keepcost - (int64_t)prev.keepcost);
  }
  return curr;
}
}  // namespace

AndroidMallocLogger::AndroidMallocLogger(std::ostringstream *oss,
                                         const std::string &name) :
  oss_(oss), name_(name) {
  prev_ = mallinfo();
}

AndroidMallocLogger::~AndroidMallocLogger() {
  struct mallinfo curr = mallinfo();
  LogMallinfoChange(oss_, name_, curr, prev_);
}

}  // namespace compat
}  // namespace deepvan
