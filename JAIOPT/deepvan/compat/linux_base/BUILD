package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "compat_linux_base",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//deepvan/compat:compat_base",
        "//deepvan/compat/posix:compat_posix",
    ],
    alwayslink = 1,
)
