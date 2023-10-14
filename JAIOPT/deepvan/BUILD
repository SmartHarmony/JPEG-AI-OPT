config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux",
    define_values = {
        "linux": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    define_values = {
        "darwin": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armv7",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm_linux_aarch64",
    values = {
        "crosstool_top": "//scripts/crosstools/aarch64_compiler:toolchain",
        "cpu": "aarch64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm_linux_armhf",
    values = {
        "crosstool_top": "//scripts/crosstools/arm_compiler:toolchain",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "neon_support",
    define_values = {
        "neon": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "hexagon_support",
    define_values = {
        "hexagon": "true",
    },
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "hta_support",
    define_values = {
        "hta": "true",
    },
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "openmp_support",
    define_values = {
        "openmp": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "static_openmp_support",
    define_values = {
        "static_openmp": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "opencl_support",
    define_values = {
        "opencl": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "quantize_enabled",
    define_values = {
        "quantize": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "memprof_support",
    define_values = {
        "memprof": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fallback_support",
    define_values = {
        "fallback": "true",
    },
    visibility = ["//visibility:public"],
)
