#ifndef DEEPVAN_CORE_RUNTIME_OPENCL_CL2_HEADER_H_
#define DEEPVAN_CORE_RUNTIME_OPENCL_CL2_HEADER_H_

// Do not include cl2.hpp directly, include this header instead.

#include "deepvan/compat/port-arch.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#ifdef DEEPVAN_OS_MAC
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_TARGET_OPENCL_VERSION 200
#endif  // DEEPVAN_OS_MAC

#ifdef DEEPVAN_OS_MAC
// disable deprecated warning in macOS 10.14
#define CL_SILENCE_DEPRECATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif  // DEEPVAN_OS_MAC

#include "include/CL/cl2.hpp"

#ifdef DEEPVAN_OS_MAC
#pragma GCC diagnostic pop
#endif

#endif  // DEEPVAN_CORE_RUNTIME_OPENCL_CL2_HEADER_H_
