package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//deepvan:deepvan.bzl",
    "is_memprof_support",
)

cc_library(
    name = "export",
    hdrs = [
        "deepvan.h",
    ],
    srcs = [
        "status.cc",
    ],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"] + is_memprof_support([
	"-DMEMPROF_SUPPORT",
    ]),
    deps = 
    [
        "//deepvan/proto:deepvan_cc",
    ] + is_memprof_support([
        "//deepvan/tools:memprof_support",
        "//deepvan/tools:memprof_support_hdrs",
    ])
)

exports_files(["xgen.h"])
