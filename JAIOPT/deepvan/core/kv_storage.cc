#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <climits>
#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

#include "deepvan/core/kv_storage.h"
#include "deepvan/utils/macros.h"
#include "deepvan/utils/logging.h"

namespace deepvan {
namespace {
void ParseKVData(const unsigned char *data,
                 size_t data_size,
                 std::map<std::string, std::vector<unsigned char>> *kv_map) {
  const size_t int_size = sizeof(int32_t);

  size_t parsed_offset = 0;
  int64_t num_tuple = 0;
  memcpy(&num_tuple, data, sizeof(num_tuple));
  data += sizeof(num_tuple);
  parsed_offset += sizeof(num_tuple);
  int32_t key_size = 0;
  int32_t value_size = 0;
  for (int i = 0; i < num_tuple; ++i) {
    memcpy(&key_size, data, int_size);
    data += int_size;
    std::unique_ptr<char[]> key(new char[key_size+1]);
    memcpy(&key[0], data, key_size);
    data += key_size;
    key[key_size] = '\0';
    parsed_offset += int_size + key_size;

    memcpy(&value_size, data, int_size);
    data += int_size;
    std::vector<unsigned char> value(value_size);
    memcpy(value.data(), data, value_size);
    data += value_size;
    parsed_offset += int_size + value_size;
    CONDITIONS(parsed_offset <= data_size,
               "Paring storage data out of range: ",
               parsed_offset, " > ", data_size);

    kv_map->emplace(std::string(&key[0]), value);
  }
}

}  // namespace

class FileStorageFactory::Impl {
 public:
  explicit Impl(const std::string &path);

  std::shared_ptr<KVStorage> CreateStorage(const std::string &name);

 private:
  std::string path_;
};

FileStorageFactory::Impl::Impl(const std::string &path): path_(path) {}

std::shared_ptr<KVStorage> FileStorageFactory::Impl::CreateStorage(
    const std::string &name) {
  return std::shared_ptr<KVStorage>(new FileStorage(path_ + "/" + name));
}

FileStorageFactory::FileStorageFactory(const std::string &path):
    impl_(new FileStorageFactory::Impl(path)) {}

FileStorageFactory::~FileStorageFactory() = default;

std::shared_ptr<KVStorage> FileStorageFactory::CreateStorage(
    const std::string &name) {
  return impl_->CreateStorage(name);
}

FileStorage::FileStorage(const std::string &file_path):
    loaded_(false), data_changed_(false), file_path_(file_path) {}

int FileStorage::Load() {
  struct stat st;
  if (stat(file_path_.c_str(), &st) == -1) {
    if (errno == ENOENT) {
      VLOG(1) << "File " << file_path_
              << " does not exist";
      return 0;
    } else {
      LOG(WARNING) << "Stat file " << file_path_
                   << " failed, error code: " << strerror(errno);
      return -1;
    }
  }
  utils::WriteLock lock(&data_mutex_);
  if (loaded_) {
    return 0;
  }
  int fd = open(file_path_.c_str(), O_RDONLY);
  if (fd < 0) {
    if (errno == ENOENT) {
      LOG(INFO) << "File " << file_path_
                << " does not exist";
      return 0;
    } else {
      LOG(WARNING) << "open file " << file_path_
                   << " failed, error code: " << strerror(errno);
      return -1;
    }
  }
  size_t file_size = st.st_size;
  unsigned char *file_data =
    static_cast<unsigned char *>(mmap(nullptr, file_size, PROT_READ,
          MAP_PRIVATE, fd, 0));
  int res = 0;
  if (file_data == MAP_FAILED) {
    LOG(WARNING) << "mmap file " << file_path_
                 << " failed, error code: " << strerror(errno);

    res = close(fd);
    if (res != 0) {
      LOG(WARNING) << "close file " << file_path_
                   << " failed, error code: " << strerror(errno);
    }
    return -1;
  }

  ParseKVData(file_data, file_size, &data_);
  res = munmap(file_data, file_size);
  if (res != 0) {
    LOG(WARNING) << "munmap file " << file_path_
                 << " failed, error code: " << strerror(errno);
    res = close(fd);
    if (res != 0) {
      LOG(WARNING) << "close file " << file_path_
                   << " failed, error code: " << strerror(errno);
    }
    return -1;
  }
  res = close(fd);
  if (res != 0) {
    LOG(WARNING) << "close file " << file_path_
                 << " failed, error code: " << strerror(errno);
    return -1;
  }
  loaded_ = true;
  return 0;
}

bool FileStorage::Clear() {
  utils::WriteLock lock(&data_mutex_);
  if (!data_.empty()) {
    data_.clear();
    data_changed_ = true;
  }
  return true;
}

bool FileStorage::Insert(const std::string &key,
                         const std::vector<unsigned char> &value) {
  utils::WriteLock lock(&data_mutex_);
  auto res = data_.emplace(key, value);
  if (!res.second) {
    data_[key] = value;
  }
  data_changed_ = true;
  return true;
}

const std::vector<unsigned char> *FileStorage::Find(const std::string &key) {
  utils::ReadLock lock(&data_mutex_);
  auto iter = data_.find(key);
  if (iter == data_.end()) return nullptr;

  return &(iter->second);
}

int FileStorage::Flush() {
  utils::WriteLock lock(&data_mutex_);
  if (!data_changed_)  return 0;
  int fd = open(file_path_.c_str(), O_WRONLY | O_CREAT, 0600);
  if (fd < 0) {
    LOG(WARNING) << "open file " << file_path_
                 << " failed, error code: " << strerror(errno);
    return -1;
  }

  const size_t int_size = sizeof(int32_t);

  int64_t data_size = sizeof(int64_t);
  for (auto &kv : data_) {
    data_size += 2 * int_size + kv.first.size() + kv.second.size();
  }
  std::unique_ptr<unsigned char[]> buffer(new unsigned char[data_size]);
  unsigned char *buffer_ptr = &buffer[0];

  int64_t num_of_data = data_.size();
  memcpy(buffer_ptr, &num_of_data, sizeof(int64_t));
  buffer_ptr += sizeof(int64_t);
  for (auto &kv : data_) {
    int32_t key_size = kv.first.size();
    memcpy(buffer_ptr, &key_size, int_size);
    buffer_ptr += int_size;

    memcpy(buffer_ptr, kv.first.c_str(), kv.first.size());
    buffer_ptr += kv.first.size();

    int32_t value_size = kv.second.size();
    memcpy(buffer_ptr, &value_size, int_size);
    buffer_ptr += int_size;

    memcpy(buffer_ptr, kv.second.data(), kv.second.size());
    buffer_ptr += kv.second.size();
  }
  int res = 0;
  buffer_ptr = &buffer[0];
  int64_t remain_size = data_size;
  while (remain_size > 0) {
    size_t buffer_size = std::min<int64_t>(remain_size, SSIZE_MAX);
    res = write(fd, buffer_ptr, buffer_size);
    if (res == -1) {
      LOG(WARNING) << "write file " << file_path_
                   << " failed, error code: " << strerror(errno);
      res = close(fd);
      if (res != 0) {
        LOG(WARNING) << "close file " << file_path_
                     << " failed, error code: " << strerror(errno);
      }
      return -1;
    }
    remain_size -= buffer_size;
    buffer_ptr += buffer_size;
  }

  res = close(fd);
  if (res != 0) {
    LOG(WARNING) << "close file " << file_path_
                 << " failed, error code: " << strerror(errno);
    return -1;
  }
  data_changed_ = false;
  return 0;
}


ReadOnlyByteStreamStorage::ReadOnlyByteStreamStorage(
    const unsigned char *byte_stream, size_t byte_stream_size) {
  ParseKVData(byte_stream, byte_stream_size, &data_);
}

int ReadOnlyByteStreamStorage::Load() {
  return 0;
}

bool ReadOnlyByteStreamStorage::Clear() {
  LOG(FATAL) << "ReadOnlyByteStreamStorage should not clear data";
  return true;
}

const std::vector<unsigned char>* ReadOnlyByteStreamStorage::Find(
    const std::string &key) {
  auto iter = data_.find(key);
  if (iter == data_.end()) return nullptr;

  return &(iter->second);
}

bool ReadOnlyByteStreamStorage::Insert(
    const std::string &key,
    const std::vector<unsigned char> &value) {
  UNUSED_VARIABLE(key);
  UNUSED_VARIABLE(value);
  LOG(FATAL) << "ReadOnlyByteStreamStorage should not insert data";
  return true;
}

int ReadOnlyByteStreamStorage::Flush() {
  return 0;
}

};  // namespace deepvan
