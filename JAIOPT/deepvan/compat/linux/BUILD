package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//deepvan:deepvan.bzl", "is_linux")

cc_library(
    name = "compat_linux",
    srcs = is_linux(glob([
        "*.cc",
    ])),
    hdrs = is_linux(glob([
        "*.h",
    ])),
    deps = is_linux([
        "//deepvan/compat/linux_base:compat_linux_base",
    ]),
    alwayslink = 1,
)
