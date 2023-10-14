package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//deepvan:deepvan.bzl", "is_android")

cc_library(
    name = "compat_android",
    srcs = is_android(glob([
        "*.cc",
    ])),
    hdrs = is_android(glob([
        "*.h",
    ])),
    deps = is_android([
        "//deepvan/compat/linux_base:compat_linux_base",
    ]),
    alwayslink = 1,
)
