# Benchmark
# Examples
load(
    "//deepvan:deepvan.bzl",
    "is_hexagon_support",
    "is_openmp_support",
    "is_static_openmp_support",
    "is_android",
    "is_opencl_support",
    "is_darwin",
    "is_fallback_support",
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "statistics",
    srcs = ["statistics.cc"],
    hdrs = ["statistics.h"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    visibility = ["//visibility:public"],
    deps = [
        "//deepvan/utils",
    ],
)

cc_binary(
    name = "benchmark_model_static",
    srcs = [
        "benchmark_model.cc",
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_opencl_support(["-DOPENCL_SUPPORT"]),
    linkopts = is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 1,
    deps = [
        ":statistics",
        "//external:gflags_nothreads",
        "//deepvan/codegen:libmodels",
        "//deepvan/codegen:generated_deepvan_engine_factory",
        "//deepvan/executor:executor",
        # ":lib_benchmark_opencl",
    ],
)

cc_binary(
    name = "benchmark_model_dynamic",
    srcs = [
        "benchmark_model.cc",
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_android(["-DOPENCL_SUPPORT"]),
    linkopts = [
        "-lm",
    ] + is_openmp_support([
        "-fopenmp"
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]) + is_android([
        "-ldl",
        "-pie",
        "-llog",
    ]),
    linkstatic = 0,
    deps = [
        ":statistics",
        "//external:gflags_nothreads",
        "//deepvan/codegen:generated_deepvan_engine_factory",
        "//deepvan/codegen:libmodels",
        "//deepvan/executor:executor_dynamic",
        # ":lib_benchmark_opencl",
    ],
)


cc_library(
    name = "lib_benchmark_opencl",
    srcs = glob(
        ["bench_opencl/*.cc",]
    ),
    hdrs = [
        "bench_opencl/benchmark_opencl.h"
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + is_opencl_support(["-DOPENCL_SUPPORT"]),
    deps = [
        ":statistics",
        "//external:gflags_nothreads",
        "//deepvan/codegen:libmodels",
        "//deepvan/codegen:generated_deepvan_engine_factory",
        "//deepvan/executor:executor",
    ],
)

cc_library(
    name = "executor_merged",
    srcs = [
        "executor_merged.a",
    ],
    visibility = ["//visibility:private"],
)

cc_binary(
    name = "model_throughput_test",
    srcs = ["model_throughput_test.cc"],
    linkopts = is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 1,
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        ":executor_merged",
        "//external:gflags_nothreads",
        "//deepvan/core",
    ],
)

cc_binary(
    name = "deepvan_run_static",
    srcs = ["deepvan_run.cc"],
    copts = [
        "-Werror",
        "-Wextra",
    ] + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_fallback_support([
        "-DFALLBACK_SUPPORT",
    ]),
    linkopts = is_darwin(
        [],
        default_value = ["-fuse-ld=gold"],
    ) + is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//deepvan/codegen:generated_deepvan_engine_factory",
        "//deepvan/codegen:libmodels",
        "//deepvan/executor",
    ] + is_fallback_support([
        "//deepvan/executor:xgen_internal",
        "@org_tensorflow//tensorflow/lite/delegates/nnapi/java/src/main/native",
        "@org_tensorflow//tensorflow/lite/java/src/main/native",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/java/src/main/native",
    ]),
)

cc_import(
    name = "lib_executor_dynamic",
    shared_library = "//deepvan/executor:libexecutor_shared.so",
)

cc_binary(
    name = "deepvan_run_dynamic",
    srcs = ["deepvan_run.cc"],
    copts = [
        "-Werror",
        "-Wextra",
    ] + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]),
    linkopts = is_darwin(
        [],
        default_value = ["-fuse-ld=gold"],
    ) + is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 0,
    deps = [
        "//external:gflags_nothreads",
        "//deepvan/codegen:generated_deepvan_engine_factory",
        "//deepvan/codegen:libmodels",
        "//deepvan/executor:executor_dynamic",
        # "//deepvan/run:lib_executor_dynamic",
    ],
)

cc_binary(
    name = "xgen_run_static",
    srcs = [
        "xgen_run.cc",
        "//deepvan/export:xgen.h",
    ],
    copts = [
        "-Werror",
        "-Wextra",
    ] + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_fallback_support([
        "-DFALLBACK_SUPPORT",
    ]),
    linkopts = is_darwin(
        [],
        default_value = ["-fuse-ld=gold"],
    ) + is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//deepvan/codegen:generated_deepvan_engine_factory",
        "//deepvan/codegen:libmodels",
        "//deepvan/executor:xgen_dynamic",
    ],
)

cc_binary(
    name = "xgen_run_dl",
    srcs = [
        "xgen_run_dl.cc",
        "//deepvan/export:xgen.h",
    ],
    copts = [
        "-Werror",
        "-Wextra",
    ] + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_fallback_support([
        "-DFALLBACK_SUPPORT",
    ]),
    linkopts = is_darwin(
        [],
        default_value = ["-fuse-ld=gold"],
    ) + is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//deepvan/codegen:libmodels",
    ],
)
