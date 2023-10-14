package(
    default_visibility=["//visibility:public"],
)


py_library(
    name="net_builder",
    srcs=[
        "net_builder/net_builder.py",
    ],
    deps=[
        ":converter_lib",
        ":model_persistent_lib",
    ],
    srcs_version="PY2AND3",
)

py_binary(
    name="net_builder_test",
    srcs=[
        "net_builder/net_builder_test.py",
    ],
    deps=[
        ":net_builder",
    ],
    srcs_version="PY2AND3",
)

py_library(
    name="converter_lib",
    srcs=[
        "net_converter/base_converter.py",
        "net_converter/onnx_converter.py",
        "net_converter/convert_util.py",
        "net_converter/transformer.py",

    ],
    srcs_version="PY2AND3",
    deps=[
        "//deepvan/proto:deepvan_py",
    ],
)

py_library(
    name="model_persistent_lib",
    srcs=[
        "tools/model_persistent.py",
    ],
    srcs_version="PY2AND3",
    deps=[
        "//deepvan/proto:deepvan_py",
    ],
)

py_binary(
    name="converter",
    srcs=[
        "tools/converter.py"
    ],
    srcs_version="PY2AND3",
    deps=[
        ":converter_lib",
        ":model_persistent_lib",
        "@six_archive//:six",
    ],
)

py_binary(
    name="layers_validate",
    srcs=[
        "tools/layers_validate.py",
    ],
    srcs_version="PY2AND3",
    deps=[
        ":converter_lib",
        ":model_persistent_lib",
    ],
)


py_binary(
    name="check_tensors",
    srcs=[
        "validate/check_tensors.py"
    ],
    srcs_version="PY2AND3",
    deps=[
        "//deepvan/proto:deepvan_py",
        ":converter_lib"
    ],
)
