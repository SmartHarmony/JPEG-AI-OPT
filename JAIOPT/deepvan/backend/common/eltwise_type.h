#ifndef DEEPVAN_BACKEND_COMMON_ELTWISE_TYPE_H_
#define DEEPVAN_BACKEND_COMMON_ELTWISE_TYPE_H_

namespace deepvan {

enum EltwiseType {
  SUM = 0,
  SUB = 1,
  PROD = 2,
  DIV = 3,
  MIN = 4,
  MAX = 5,
  NEG = 6,
  ABS = 7,
  SQR_DIFF = 8,
  POW = 9,
  EQUAL = 10,
  FLOOR_DIV = 11,
  NONE = 12,
  EXP = 14,
  ERF = 15,
};

}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_COMMON_ELTWISE_TYPE_H_
