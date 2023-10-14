# Description:
# Generated model and runtime code.
#
package(
    default_visibility = ["//visibility:public"],
)

load("//deepvan:deepvan.bzl", "encrypt_opencl_kernel_genrule")

cc_library(
    name = "libmodels",
    srcs = glob(["models/*/*.cc"]),
    hdrs = glob(["models/*/*.h"]),
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//deepvan/core",
    ],
)

genrule(
    name = "generated_models",
    srcs = [
        ":libmodels",
    ],
    outs = ["libgenerated_models.a"],
    cmd = "binary=`echo $(locations :libmodels) | xargs -n 1 | grep libmodels.a`;" +
          "$(STRIP) -x $$binary -o $@",
    visibility = ["//visibility:public"],
)

encrypt_opencl_kernel_genrule()

cc_library(
    name = "generated_opencl",
    srcs = ["opencl/encrypt_opencl_kernel.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_opencl_binary",
    srcs = ["opencl/opencl_binary.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_opencl_parameter",
    srcs = ["opencl/opencl_parameter.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_deepvan_engine_factory",
    hdrs = glob(["engine/*.h"]),
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//deepvan/export",
    ],
)

cc_library(
  name = "generated_executor",
  srcs = glob(["lib/*"]),
  linkstatic = 1,
  visibility = ["//visibility:public"],
)
