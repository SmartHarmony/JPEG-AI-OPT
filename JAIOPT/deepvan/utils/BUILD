# Description:
# Deepvan utils.
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
    "is_openmp_support",
)

cc_library(
    name = "utils_hdrs",
    hdrs = glob([
        "*.h",
    ]),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ],
    deps = [
        "//deepvan/compat:compat_api",
        "//deepvan/export",
    ],
)

cc_library(
    name = "utils",
    srcs = glob(
        [
            "*.cc",
        ],
        exclude = [
            "*_test.cc",
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
    ]),
    linkopts = is_android([
        "-llog",
    ]),
    deps = [
        ":utils_hdrs",
    ],
    alwayslink = 1,
)

cc_test(
    name = "utils_test",
    testonly = 1,
    srcs = glob(
        [
            "*_test.cc",
        ],
    ),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ],
    linkstatic = 1,
    deps = [
        ":utils",
        "//deepvan/compat",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
