# Description:
# Deepvan executor.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//deepvan:deepvan.bzl",
    "is_android",
    "is_android_armv7",
    "is_darwin",
    "is_hexagon_support",
    "is_hta_support",
    "is_linux",
    "is_neon_support",
    "is_opencl_support",
    "is_openmp_support",
    "is_static_openmp_support",
    "is_memprof_support",
    "is_fallback_support",
)

# Base shared/static library
cc_library(
    name = "executor",
    srcs = [
        "capability.cc",
        "deepvan.cc",
    ],
    copts = [
        "-Werror",
        "-Wextra",
    ] + is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
    ]) + is_android_armv7([
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]) + is_hta_support([
        "-DHTA_SUPPORT",
    ]) + is_memprof_support([
        "-DMEMPROF_SUPPORT",
    ]),
    deps = [
        "//deepvan/backend",
        "//deepvan/export",
    ],
    alwayslink = 1,
)

# Shared library binary, which can load directly
cc_binary(
    name = "libexecutor_shared.so",
    linkopts = is_darwin(
        ["-Wl,-install_name,libexecutor_shared.so"],
        [
            "-Wl,-soname,libexecutor_shared.so",
            "-Wl,--version-script",
            "$(location //deepvan/executor:deepvan_version_script.lds)",
        ],
    ) + is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]) + is_memprof_support([
        "-DMEMPROF_SUPPORT",
    ]),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//deepvan/executor",
        "//deepvan/executor:deepvan_version_script.lds",
    ],
)

exports_files(
    [
        "deepvan_version_script.lds",
        "xgen.lds",
    ],
)

# standard alone dynamic library
cc_library(
    name = "executor_dynamic",
    srcs = ["libexecutor_shared.so"],
    visibility = ["//visibility:public"],
)

# standard alone static library
# For details, see https://github.com/bazelbuild/bazel/issues/5200
genrule(
    name = "executor_static",
    srcs = [
        "//deepvan/core",
        "//deepvan/backend:common",
        "//deepvan/backend:ref_kernels",
        "//deepvan/backend:internal_ops",
        "//deepvan/backend",
        "//deepvan/executor",
        "//deepvan/compat:compat_base",
        "//deepvan/compat/posix:compat_posix",
        "//deepvan/export",
        "//deepvan/utils",
        "//deepvan/proto:deepvan_cc",
        "//deepvan/compat/linux_base:compat_linux_base",
        "@com_google_protobuf_dep//:protobuf_lite",
    ] + is_android([
        "//deepvan/compat/android:compat_android",
    ]) + is_linux([
        "//deepvan/compat/linux:compat_linux",
    ]) + is_darwin([
        "//deepvan/compat/darwin:compat_darwin",
        "//deepvan/compat/darwin:darwin_ar_merge",
    ]) + is_opencl_support([
        "//deepvan/backend:opencl_kernels",
        "//deepvan/codegen:generated_opencl",
    ]) + is_neon_support([
        "//deepvan/backend:arm_neon_kernels",
    ]),
    outs = ["executor.a"],
    cmd = "tmp_mri_file=mktempdeepvan-static-lib-mri.XXXXXXXXXX;" +
          "mri_stream=$$(python $(location //lothar:archive_static_lib) " +
          "$(locations //deepvan/core:core) " +
          "$(locations //deepvan/backend:common) " +
          "$(locations //deepvan/backend:ref_kernels) " +
          is_neon_support(
              "$(locations //deepvan/backend:arm_neon_kernels) ",
              default_value = "",
          ) +
          is_opencl_support(
              "$(locations //deepvan/backend:opencl_kernels) ",
              default_value = "",
          ) +
          "$(locations //deepvan/backend:internal_ops) " +
          "$(locations //deepvan/backend:backend) " +
          "$(locations //deepvan/executor:executor) " +
          "$(locations //deepvan/compat:compat_base) " +
          "$(locations //deepvan/compat/posix:compat_posix) " +
          is_android(
              "$(locations //deepvan/compat/linux_base:compat_linux_base) " +
              "$(locations //deepvan/compat/android:compat_android) ",
              default_value = "",
          ) +
          is_linux(
              "$(locations //deepvan/compat/linux_base:compat_linux_base) " +
              "$(locations //deepvan/compat/linux:compat_linux) ",
              default_value = "",
          ) +
          is_darwin(
              "$(locations //deepvan/compat/darwin:compat_darwin) ",
              default_value = "",
          ) +
          "$(locations //deepvan/export:export) " +
          "$(locations //deepvan/utils:utils) " +
          "$(locations //deepvan/proto:deepvan_cc) " +
          "$(locations @com_google_protobuf_dep//:protobuf_lite) " +
          is_opencl_support(
              "$(locations //deepvan/codegen:generated_opencl) ",
              default_value = "",
          ) +
          "$@ " +
          "$$tmp_mri_file);" +
          is_darwin(
              "bash $(locations //deepvan/compat/darwin:darwin_ar_merge) $(AR) < $$tmp_mri_file;",
            #  TODO @vgod fixme here  
            # using /Users/newway/Library/Android/android-ndk-r16b/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/bin/aarch64-linux-android-ar
            # when you want to build a static lib
              default_value = "$(AR) -M <$$tmp_mri_file;",
          ) +
          "rm -rf $$tmp_mri_file;",
    # "$(STRIP) -x $@;",  # FIXME this will crash
    tools = ["//lothar:archive_static_lib"],
    visibility = ["//visibility:public"],
)

# XGen
cc_library(
    name = "xgen_internal",
    srcs = ["xgen.cc"],
    hdrs = ["//deepvan/export:xgen.h"],
    copts = [
        "-Werror",
        "-Wextra",
    ] + is_openmp_support(["-fopenmp", "-static-openmp"]) + is_neon_support([
        "-DNEON_SUPPORT",
    ]) + is_android_armv7([
        "-mfpu=neon",
    ]) + is_android_armv7([
        "-mfloat-abi=softfp",
    ]) + is_opencl_support([
        "-DOPENCL_SUPPORT",
    ]) + is_hexagon_support([
        "-DHEXAGON_SUPPORT",
    ]) + is_hta_support([
        "-DHTA_SUPPORT",
    ]) + is_memprof_support([
        "-DMEMPROF_SUPPORT",
    ]) + is_fallback_support([
        "-DFALLBACK_SUPPORT",
    ]),
    deps = [
        "//deepvan/executor",
    ] + is_fallback_support([
        "@org_tensorflow//tensorflow/lite/delegates/nnapi/java/src/main/native",
        "@org_tensorflow//tensorflow/lite/java/src/main/native",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/java/src/main/native",
    ]),
    alwayslink = 1,
)

cc_binary(
    name = "libxgen.so",
    linkopts = is_darwin(
        ["-Wl,-install_name,libxgen.so"],
        [
            "-Wl,-soname,libxgen.so",
            "-Wl,--version-script",
            "$(location //deepvan/executor:xgen.lds)",
        ],
    ) + is_openmp_support([
        "-fopenmp",
    ]) + is_static_openmp_support([
        "-static-openmp",
    ]) + is_memprof_support([
        "-DMEMPROF_SUPPORT",
    ]),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//deepvan/executor:xgen_internal",
        "//deepvan/executor:xgen.lds",
    ],
)

# standard alone dynamic library
cc_library(
    name = "xgen_dynamic",
    srcs = ["libxgen.so"],
    visibility = ["//visibility:public"],
)
