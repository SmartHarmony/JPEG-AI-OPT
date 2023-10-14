package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "compat",
    deps = [
        "//deepvan/compat/android:compat_android",
        "//deepvan/compat/darwin:compat_darwin",
        "//deepvan/compat/linux:compat_linux",
    ],
)

cc_library(
    name = "compat_api",
    hdrs = [
        "env.h",
        "file_system.h",
        "logger.h",
        "port.h",
        "port-arch.h",
    ],
    deps = [
        "//deepvan/export",
    ],
)

cc_library(
    name = "compat_base",
    srcs = [
        "env.cc",
        "logger.cc",
    ],
    deps = [
        ":compat_api",
        "//deepvan/utils",
    ],
)

cc_test(
    name = "compat_test",
    testonly = 1,
    srcs = glob([
        "*_test.cc",
    ]),
    linkstatic = 1,
    deps = [
        ":compat",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)
