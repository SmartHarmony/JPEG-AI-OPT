#ifndef DEEPVAN_COMPAT_LOGGER_H_
#define DEEPVAN_COMPAT_LOGGER_H_

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace deepvan {
enum LogLevel {
  INVALID_MIN = 0,
  INFO        = 1,
  WARNING     = 2,
  ERROR       = 3,
  FATAL       = 4,
  INVALID_MAX,
};

namespace compat {

inline bool LogLevelPassThreashold(const LogLevel level,
                                   const LogLevel threshold) {
  return level >= threshold;
}

LogLevel LogLevelFromStr(const char *log_level_str);
int VLogLevelFromStr(const char *vlog_level_str);

inline LogLevel MinLogLevelFromEnv() {
  // Read the min log level from env once during the first call to logging.
  static LogLevel log_level = LogLevelFromStr(getenv("CPP_MIN_LOG_LEVEL"));
  return log_level;
}

inline int MinVLogLevelFromEnv() {
  // Read the min vlog level from env once during the first call to logging.
  static int vlog_level = VLogLevelFromStr(getenv("DEEPVAN_CPP_MIN_VLOG_LEVEL"));
  return vlog_level;
}

class LogWriter {
 public:
  LogWriter() = default;
  virtual ~LogWriter() = default;
  virtual void WriteLogMessage(const char *fname,
                               const int line,
                               const LogLevel severity,
                               const char *message);
};

class Logger : public std::ostringstream {
 public:
  Logger(const char *fname, int line, LogLevel severity);
  ~Logger();

 private:
  void GenerateLogMessage();
  void DealWithFatal();

  const char *fname_;
  int line_;
  LogLevel severity_;
};

}  // namespace compat

// Whether the log level pass the env configured threshold, can be used for
// short cutting.
inline bool ShouldGenerateLogMessage(LogLevel severity) {
  LogLevel threshold = compat::MinLogLevelFromEnv();
  return compat::LogLevelPassThreashold(severity, threshold);
}

inline bool ShouldGenerateVLogMessage(int vlog_level) {
  int threshold = compat::MinVLogLevelFromEnv();
  return ShouldGenerateLogMessage(INFO) &&
         vlog_level <= threshold;
}
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_LOGGER_H_
