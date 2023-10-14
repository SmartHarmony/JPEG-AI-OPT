#ifndef DEEPVAN_COMPAT_PORT_ARCH_H_
#define DEEPVAN_COMPAT_PORT_ARCH_H_

#if defined __APPLE__
# define DEEPVAN_OS_MAC 1
# if TARGET_OS_IPHONE
#  define OS__IOS 1
# endif
#elif defined __linux__
# define OS__LINUX 1
# if defined(__ANDROID__) || defined(ANDROID)
#  define OS__LINUX_ANDROID 1
# endif
#endif

#endif  // DEEPVAN_COMPAT_PORT_ARCH_H_
