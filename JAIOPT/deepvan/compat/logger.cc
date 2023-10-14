#include "deepvan/compat/logger.h"

#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>

#include "deepvan/compat/env.h"
#include "deepvan/utils/string_util.h"

namespace deepvan {
namespace compat {

inline bool IsValidLogLevel(const LogLevel level) {
  return level > LogLevel::INVALID_MIN &&
         level < LogLevel::INVALID_MAX;
}

LogLevel LogLevelFromStr(const char *log_level_str) {
  if (log_level_str != nullptr) {
    std::string ls = ToUpper(log_level_str);

    if (ls == "I" || ls == "INFO") {
      return LogLevel::INFO;
    }
    if (ls == "W" || ls == "WARNING") {
      return LogLevel::WARNING;
    }
    if (ls == "E" || ls == "ERROR") {
      return LogLevel::ERROR;
    }
    if (ls == "F" || ls == "FATAL") {
      return LogLevel::FATAL;
    }
  }

  return LogLevel::INVALID_MIN;
}

char LogLevelToShortStr(LogLevel level) {
  if (!IsValidLogLevel(level)) {
    level = LogLevel::INFO;
  }

  return "IWEF"[static_cast<int>(level) - 1];
}

int VLogLevelFromStr(const char *vlog_level_str) {
  if (vlog_level_str != nullptr) {
    return atoi(vlog_level_str);
  }

  return 0;
}


void LogWriter::WriteLogMessage(const char *fname,
                                const int line,
                                const LogLevel severity,
                                const char *message) {
  printf("%c %s:%d] %s\n", LogLevelToShortStr(severity), fname, line, message);
}

Logger::Logger(const char *fname, int line, LogLevel severity)
    : fname_(fname), line_(line), severity_(severity) {}

void Logger::GenerateLogMessage() {
  LogWriter *log_writer = Env::Default()->GetLogWriter();
  log_writer->WriteLogMessage(fname_, line_, severity_, str().c_str());

  // When there is a fatal log, terminate execution
  if (severity_ == LogLevel::FATAL) {
    DealWithFatal();
  }
}

void Logger::DealWithFatal() {
  // When there is a fatal log, log the backtrace and abort.
  LogWriter *log_writer = Env::Default()->GetLogWriter();
  std::vector<std::string> bt = Env::Default()->GetBackTraceUnsafe(50);
  if (!bt.empty()) {
    log_writer->WriteLogMessage(fname_, line_, severity_, "backtrace:");
    for (size_t i = 0; i < bt.size(); ++i) {
      std::ostringstream os;
      os << " " << bt[i];
      log_writer->WriteLogMessage(fname_, line_, severity_, os.str().c_str());
    }
  }

  exit(-1);
}

Logger::~Logger() {
  static const LogLevel min_log_level = MinLogLevelFromEnv();
  if (LogLevelPassThreashold(severity_, min_log_level)) {
    GenerateLogMessage();
  }
}

}  // namespace compat
}  // namespace deepvan
