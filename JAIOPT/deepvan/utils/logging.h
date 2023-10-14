#ifndef DEEPVAN_UTILS_LOGGING_H_
#define DEEPVAN_UTILS_LOGGING_H_

#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "deepvan/compat/env.h"
#include "deepvan/compat/logger.h"
#include "deepvan/utils/macros.h"
#include "deepvan/utils/memory.h"
#include "deepvan/utils/string_util.h"

namespace deepvan {
const std::string DEBUG_GPU = "debug_gpu\t";
const int DEBUG = 4;
namespace logging_internal {

#define LOG(severity) \
  ::deepvan::compat::Logger(__FILE__, __LINE__, deepvan::severity)

#define LOG_PTR(severity) \
  make_unique<deepvan::compat::Logger>(__FILE__, __LINE__, deepvan::severity)

#define VLOG_IS_ON(vll) (deepvan::ShouldGenerateVLogMessage(vll))
#define VLOG(vll) if (VLOG_IS_ON(vll)) LOG(INFO)

// DEEPVAN_CHECK dies with a fatal error if condition is not true.
// DEEPVAN_ASSERT is controlled by NDEBUG ('-c opt' for bazel) while CONDITIONS
// will be executed regardless of compilation mode.
// Therefore, it is safe to do things like:
//    CONDITIONS(fp->Write(x) == 4)
//    CONDITIONS(fp->Write(x) == 4, "Write failed")
#define CONDITIONS(condition, ...) \
  if (!(condition)) \
  LOG(FATAL) << "Check failed: " #condition " " << deepvan::MakeString(__VA_ARGS__)

#define CONDITIONS_OP(condition, op) \
  if (!(condition)) \
  UNSUPPORTED_OP(op)

template <typename T>
T &&CheckNotNull(const char *file, int line, const char *exprtext, T &&t) {
  if (t == nullptr) {
    ::deepvan::compat::Logger(file, line, FATAL) << std::string(exprtext);
  }
  return std::forward<T>(t);
}

#define CONDITIONS_NOTNULL(val) \
  ::deepvan::logging_internal::CheckNotNull(__FILE__, __LINE__, \
                                         "'" #val "' Must not be NULL", (val))

//DeepVan NOT IMPLEMENTED
#define STUB CONDITIONS(false, "not implemented")

#define UNSUPPORTED_OP(op) LOG(FATAL) << "[UNSUPPORTED] \"" << op << "\" is not supported according to its properties. "

#define RETURN_IF_ERROR(stmt)                           \
  {                                                          \
    VanState status = (stmt);                              \
    if (status != VanState::SUCCEED) {                \
      VLOG(0) << #stmt << " failed with error: "             \
              << status.information();                       \
      return status;                                         \
    }                                                        \
  }

class LatencyLogger {
 public:
  LatencyLogger(int vlog_level, const std::string &message)
      : vlog_level_(vlog_level), message_(message) {
    if (VLOG_IS_ON(vlog_level_)) {
      start_micros_ = NowMicros();
      VLOG(vlog_level_) << message_ << " started";
    }
  }
  ~LatencyLogger() {
    if (VLOG_IS_ON(vlog_level_)) {
      int64_t stop_micros = NowMicros();
      VLOG(vlog_level_) << message_
                        << " latency: " << stop_micros - start_micros_ << " us";
    }
  }

 private:
  const int vlog_level_;
  const std::string message_;
  int64_t start_micros_;

  DISABLE_COPY_AND_ASSIGN(LatencyLogger);
};

#define LATENCY_LOGGER(vlog_level, ...)                                  \
  deepvan::logging_internal::LatencyLogger latency_logger_##__line__(            \
      vlog_level, VLOG_IS_ON(vlog_level) ? deepvan::MakeString(__VA_ARGS__) : "")


#ifdef ENABLE_MALLOC_LOGGING
#define MEMORY_LOGGING_GUARD()                                      \
  auto malloc_logger_##__line__ = compat::Env::Default()->NewMallocLogger( \
      ::deepvan::compat::Logger(__FILE__, __LINE__, deepvan::INFO), \
      std::string(__FILE__) + ":" + std::string(__func__));
#else
#define MEMORY_LOGGING_GUARD()
#endif

}  // namespace logging_internal
}  // namespace deepvan

#endif  // DEEPVAN_UTILS_LOGGING_H_
