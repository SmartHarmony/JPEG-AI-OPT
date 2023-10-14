# Description:
# Deepvan core.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//deepvan:deepvan.bzl",
    "is_android",
    "is_android_armv7",
    "is_hexagon_support",
    "is_hexagon_or_hta_support",
    "is_hta_support",
    "is_neon_support",
    "is_not_hexagon_support",
    "is_opencl_support",
    "is_openmp_support",
    "is_memprof_support",
)

cc_library(
    name = "core",
    srcs = glob(
        [
            "*.cc",
            "runtime/cpu/*.cc",
        ],
        exclude = [
            "*_test.cc",
        ],
    ) + is_opencl_support(glob(
        [
            "runtime/opencl/*.cc",
        ],
    )) + is_hexagon_support([
        "runtime/hexagon/hexagon_dsp_wrapper.cc",
    ]) + is_hta_support([
        "runtime/hexagon/hexagon_hta_wrapper.cc",
    ]),
    hdrs = glob([
        "*.h",
        "runtime/cpu/*.h",
    ]) + is_opencl_support(glob([
        "runtime/opencl/*.h",
    ])) + is_hexagon_or_hta_support(glob([
        "runtime/hexagon/hexagon_control_wrapper.h",
        "runtime/hexagon/hexagon_device.h",
    ])) + is_hexagon_support(glob([
        "runtime/hexagon/*dsp*.h",
    ])) + is_hta_support(glob([
        "runtime/hexagon/*hta*.h",
    ])),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
        "-DOPENMP_SUPPORT",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]) + is_hta_support([
        "-DHTA_SUPPORT",
    ]) + is_memprof_support([
        "-DMEMPROF_SUPPORT",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]),
    linkopts = ["-ldl"],
    deps = [
        "//deepvan/proto:deepvan_cc",
        "//deepvan/utils",
        "//deepvan/compat",
        "@half//:half",
    ] + is_opencl_support([
        ":opencl_headers",
        "//deepvan/codegen:generated_opencl",
    ]) + is_hexagon_support([
        "//third_party/nnlib:libhexagon",
    ]) + is_hta_support([
        "//third_party/hta",
    ]) + is_memprof_support([
	"//deepvan/tools:memprof_support",
    ]),
)

cc_library(
    name = "opencl_headers",
    hdrs = glob([
        "runtime/opencl/*.h",
    ]),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ],
    deps = [
        "@opencl_clhpp",
        "@opencl_headers_dep//:opencl20_headers",
    ],
)

cc_library(
    name = "test_benchmark_main",
    testonly = 1,
    srcs = [
        "testing/test_benchmark.cc",
        "testing/test_benchmark_result.cc",
        "testing/test_benchmark_main.cc",
    ],
    hdrs = [
        "testing/test_benchmark.h",
        "testing/test_benchmark_result.h",
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support(["-fopenmp"]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]),
    deps = [
        ":core",
        "//external:gflags_nothreads",
        "//deepvan/backend:test",
        "//deepvan/utils",
    ],
)
