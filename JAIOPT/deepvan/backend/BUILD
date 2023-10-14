# Description:
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
    "is_neon_support",
    "is_opencl_support",
    "is_openmp_support",
    "is_static_openmp_support",
)

cc_library(
    name = "common",
    srcs = glob(
        [
            "common/*.cc",
        ],
        exclude = [
            "common/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "common/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        "//deepvan/core",
    ],
)

cc_library(
    name = "testing",
    hdrs = [
        "testing/test_utils.h",
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ],
    deps = [
        "//deepvan/core",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ref_kernels",
    srcs = glob(
        [
            "ref/*.cc",
        ],
    ),
    hdrs = glob(
        [
            "ref/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
    ],
)


# After refactor, all arm neon kernels go here.
# Could be shipped to other product use.
cc_library(
    name = "arm_neon_common",
    srcs = glob(
        [
            "arm/common/*.cc",
        ],
    ),
    hdrs = glob(
        [
            "arm/common/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
    ],
)


# After refactor, all arm neon kernels go here.
# Could be shipped to other product use.
cc_library(
    name = "arm_neon_kernels",
    srcs = glob(
        [
            "arm/fp32/*.cc",
        ],
        exclude = [
            "arm/fp32/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "arm/fp32/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
        ":arm_neon_pattern_kernel",
        ":arm_neon_csr_kernel",
        ":arm_neon_slice_kernel",
        ":arm_neon_column_kernel",
        ":arm_neon_block_kernel",
    ],
)

cc_library(
    name = "arm_neon_pattern_kernel",
    srcs = glob(
        [
            "arm/pattern/*.cc",
        ],
        exclude = [
            "arm/pattern/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "arm/pattern/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
        ":arm_neon_common",
    ],
)

cc_library(
    name = "arm_neon_csr_kernel",
    srcs = glob(
        [
            "arm/csr/*.cc",
        ],
        exclude = [
            "arm/csr/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "arm/csr/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
        ":arm_neon_common",
    ],
)

cc_library(
    name = "arm_neon_slice_kernel",
    srcs = glob(
        [
            "arm/slice/*.cc",
        ],
        exclude = [
            "arm/slice/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "arm/slice/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
        ":arm_neon_common",
    ],
)

cc_library(
    name = "arm_neon_column_kernel",
    srcs = glob(
        [
            "arm/column/*.cc",
        ],
        exclude = [
            "arm/column/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "arm/column/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
        ":arm_neon_common",
    ],
)

cc_library(
    name = "arm_neon_block_kernel",
    srcs = glob(
        [
            "arm/block/*.cc",
        ],
        exclude = [
            "arm/block/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "arm/block/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
        ":arm_neon_common",
    ],
)

# After refactor, all GPU OpenCL kernels go here.
# Could be shipped to other product use.
cc_library(
    name = "opencl_kernels",
    srcs = glob(
        [
            "opencl/*.cc",
            "opencl/**/*.cc",
            "buffer_transform.cc",
            "lstm_cell.cc",
        ],
        exclude = [
            "opencl/*_test.cc",
        ],
    ),
    hdrs = glob(
        [
            "opencl/*.h",
            "opencl/**/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":common",
        "//deepvan/core",
    ],
)

cc_library(
    name = "arm_neon_kernels_test",
    srcs = glob(
        [
            "arm/fp32/*_test.cc",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":arm_neon_kernels",
        ":ref_kernels",
        ":testing",
        "@com_google_googletest//:gtest_main",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencl_kernels_test",
    srcs = glob(
        [
            "opencl/*_test.cc",
            "opencl/**/*_test.cc",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        ":opencl_kernels",
        ":ref_kernels",
        ":testing",
        "@com_google_googletest//:gtest_main",
    ],
    alwayslink = 1,
)

cc_library(
    name = "internal_ops",
    srcs = glob(
        [
            "*.cc",
            "arm/*.cc",  # remove it after refactor
        ],
        exclude = [
            "*_test.cc",
            "*_benchmark.cc",
            "ops_registry.cc",
            "ops_test_util.cc",
            "lstm_cell.cc",  # TODO: move it into opencl
            "buffer_transform.cc",  # TODO: move it into opencl
            "quantize.cc",
            "quantization_util.cc",
            "arm/*_test.cc",  # remove it after refactor
        ],
    ),
    hdrs = glob(
        [
            "*.h",
            "arm/*.h",  # remove it after refactor
        ],
        exclude = [
            "ops_registry.h",
            "ops_test_util.h",
            "fixpoint.h",
            "quantization_util.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    linkopts = is_android(["-lm"]),
    deps = [
        ":ref_kernels",
        "//deepvan/core",
    ] + is_neon_support([
        ":arm_neon_kernels",
    ]) + is_opencl_support([
        ":opencl_kernels",
    ]),
)

cc_library(
    name = "backend",
    srcs = glob(
        [
            "ops_registry.cc",
        ],
    ),
    hdrs = glob(
        [
            "ops_registry.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    linkopts = is_android(["-lm"]),
    deps = [
        "internal_ops",
    ],
)

cc_library(
    name = "test",
    testonly = 1,
    srcs = [
        "ops_test_util.cc",
    ],
    hdrs = glob([
        "*_test_util.h",
    ]),
    copts = [
        "-Werror",
        "-Wextra",
    ] + is_openmp_support(["-fopenmp"]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
    ]) + is_android_armv7([
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    deps = [
        "backend",
        "testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "ops_test",
    testonly = 1,
    srcs = glob(
        [
            "*_test.cc",
            "arm/*_test.cc",
            "common/*_test.cc",
            "ops_test_util.cc",
        ],
        exclude = [
            "fixpoint_test.cc",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    linkopts = is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 1,
    deps = [
        ":backend",
        ":test",
        "@com_google_googletest//:gtest_main",
    ] + is_neon_support([
        ":arm_neon_kernels_test",
    ]) + is_opencl_support([
        ":opencl_kernels_test",
    ]),
)

cc_test(
    name = "ops_benchmark",
    testonly = 1,
    srcs = glob(["*_benchmark.cc"]),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]),
    linkopts = is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 1,
    deps = [
        ":backend",
        "//deepvan/run:statistics",
        "//deepvan/core:test_benchmark_main",
        "//third_party/eigen3",
    ],
)
