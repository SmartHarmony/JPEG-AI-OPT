#ifndef DEEPVAN_CORE_MEMORY_OPTIMIZER_H_
#define DEEPVAN_CORE_MEMORY_OPTIMIZER_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "deepvan/core/types.h"
#include "deepvan/proto/deepvan.pb.h"

namespace deepvan {
class MemoryBlock {
public:
  inline void set_mem_id(int mem_id) { mem_id_ = mem_id; }

  inline int mem_id() const { return mem_id_; }

  inline void set_data_type(DataType data_type) { data_type_ = data_type; }

  inline DataType data_type() const { return data_type_; }

  inline void set_mem_type(MemoryType mem_type) { mem_type_ = mem_type; }

  inline MemoryType mem_type() const { return mem_type_; }

  inline void set_x(int64_t x) { x_ = x; }

  inline int64_t x() const { return x_; }

  inline void set_y(int64_t y) { y_ = y; }

  inline int64_t y() const { return y_; }

private:
  int mem_id_;
  DataType data_type_;
  MemoryType mem_type_;
  int64_t x_;
  int64_t y_;
};

class MemoryOptimizer {
public:
  struct TensorMemInfo {
    int mem_id;
    DataType data_type;
    bool has_data_format;

    TensorMemInfo(int mem_id, DataType data_type, bool has_data_format)
        : mem_id(mem_id),
          data_type(data_type),
          has_data_format(has_data_format) {}
  };

public:
  static bool IsMemoryReuseOp(const std::string &op_type);
  static bool IsMemoryReuseOp(const deepvan::OperatorProto *, const MemoryType);
  int GetTensorRefCount(const std::string &op_name);
  void UpdateTensorRef(const std::string &tensor_name);
  void UpdateTensorRef(const OperatorProto *op_def);
  void Optimize(
      const OperatorProto *op_def,
      const std::unordered_map<std::string, MemoryType> *mem_types = nullptr);

  const std::vector<MemoryBlock> &mem_blocks() const;

  const std::unordered_map<std::string, TensorMemInfo> &tensor_mem_map() const;

  std::string DebugInfo() const;

private:
  MemoryBlock CreateMemoryBlock(const OperatorProto *op_def,
                                int output_idx,
                                DataType dt,
                                MemoryType mem_type);

  MemoryBlock CreateDenseMemoryBlock(const OperatorProto *op_def,
                                     int output_idx,
                                     DataType dt,
                                     MemoryType mem_type);

private:
  std::unordered_map<std::string, int> tensor_ref_count_;
  std::vector<MemoryBlock> mem_blocks_;
  // tensor name : <mem_id, data_type>
  // Buffer Memory do not different data type, so store the data type.
  std::unordered_map<std::string, TensorMemInfo> tensor_mem_map_;
  std::unordered_map<int, int> mem_ref_count_;
  std::set<int> idle_blocks_;
};
} // namespace deepvan
#endif // DEEPVAN_CORE_MEMORY_OPTIMIZER_H_
