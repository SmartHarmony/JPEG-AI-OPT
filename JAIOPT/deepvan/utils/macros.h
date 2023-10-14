#ifndef DEEPVAN_UTILS_MACROS_H_
#define DEEPVAN_UTILS_MACROS_H_

namespace deepvan {
// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(CLASSNAME)     \
  CLASSNAME(const CLASSNAME &) = delete;            \
  CLASSNAME &operator=(const CLASSNAME &) = delete;
#endif

#ifndef DEEPVAN_EMPTY_VIRTUAL_DESTRUCTOR
#define DEEPVAN_EMPTY_VIRTUAL_DESTRUCTOR(CLASSNAME) \
 public:                                         \
  virtual ~CLASSNAME() {}
#endif

#define UNUSED_VARIABLE(var) (void)(var)
}  // namespace deepvan

#endif  // DEEPVAN_UTILS_MACROS_H_
