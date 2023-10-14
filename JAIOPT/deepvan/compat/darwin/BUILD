package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//deepvan:deepvan.bzl", "is_darwin")

cc_library(
    name = "compat_darwin",
    srcs = is_darwin(glob([
        "*.cc",
    ])),
    hdrs = is_darwin(glob([
        "*.h",
    ])),
    deps = [
        "//deepvan/compat:compat_base",
        "//deepvan/compat/posix:compat_posix",
    ],
    alwayslink = 1,
)

sh_library(
    name = "darwin_ar_merge",
    srcs = ["ar_merge_on_darwin.sh"],
)
