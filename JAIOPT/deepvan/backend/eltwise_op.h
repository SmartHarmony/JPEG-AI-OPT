#ifndef DEEPVAN_BACKEND_ELTWISE_OP_H_
#define DEEPVAN_BACKEND_ELTWISE_OP_H_

#include "deepvan/backend/common/eltwise_type.h"


namespace deepvan {

inline bool IsLogicalType(EltwiseType type) { return type == EQUAL; }

}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_ELTWISE_OP_H_
