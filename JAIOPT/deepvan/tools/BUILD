# Description:
# Deepvan memprof.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//deepvan:deepvan.bzl",
    "is_android",
    "is_android_armv7",
    "is_neon_support",
    "is_opencl_support",
    "is_openmp_support",
    "is_memprof_support",
)

cc_library(
    name = "memprof_support_hdrs",
    hdrs = glob([
        "libmemprofile/*.h",
    ]),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ],
    deps = [
        "@opencl_clhpp",
    ],
)

cc_library(
    name = "memprof_support",
    srcs = glob(
        [
            "libmemprofile/*.cc",
	    "libmemprofile/libunwind.a",
        ],
    ),
    copts = [
        "-Werror",
	"-Wno-deprecated-declarations",
      	"-shared",
	"-fPIC",
    ] + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_openmp_support([
        "-fopenmp",
    ]) + is_memprof_support([
        "-DMEMPROF_SUPPORT",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]),
    linkopts = [
        "-dl",
    ]  + is_android([
        "-llog",
    ]),
    linkstatic=1,
    deps = [
         "@opencl_headers_dep//:opencl20_headers",
         "@opencl_clhpp",
	 "memprof_support_hdrs",
    ],
)

