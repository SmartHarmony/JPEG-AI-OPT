"""Load dependencies needed to compile the library as a 3rd-party consumer."""
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bazel_related_deps():
    if "bazel_skylib" not in native.existing_rules():
        http_archive(
            name = "bazel_skylib",
            strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
            sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
            urls = ["https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz"],
        )

def protobuf_deps(root):
    """Loads common dependencies needed to compile the protobuf library."""
    if "com_google_protobuf" not in native.existing_rules():
        http_archive(
            name = "com_google_protobuf",
            strip_prefix = "protobuf-3.7.1",
            build_file = "//third_party:protobuf/protobuf.BUILD",
            sha256 = "f976a4cd3f1699b6d20c1e944ca1de6754777918320c719742e1674fcf247b7e",
            urls = [
                "https://github.com/google/protobuf/archive/v3.7.1.zip",
            ],
        )

def six_deps():
    if "six_archive" not in native.existing_rules():
        http_archive(
            name = "six_archive",
            build_file = "//third_party:six/six.BUILD",
            strip_prefix = "six-1.10.0",
            sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
            urls = [
                "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            ],
        )
        native.bind(
            name = "six",
            actual = "@six_archive//:six",
        )

def eigen_deps():
    if "eigen_dep" not in native.existing_rules():
        http_archive(
            name = "eigen_dep",
            build_file = "//third_party:eigen3/eigen.BUILD",
            sha256 = "ca7beac153d4059c02c8fc59816c82d54ea47fe58365e8aded4082ded0b820c4",
            strip_prefix = "eigen-eigen-f3a22f35b044",
            urls = [
                "http://mirror.bazel.build/bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz",
            ],
        )

def gemmlowp_deps():
    if "gemmlowp" not in native.existing_rules():
        http_archive(
            name = "gemmlowp",
            sha256 = "4e9cd60f7871ae9e06dcea5fec1a98ddf1006b32a85883480273e663f143f303",
            strip_prefix = "gemmlowp",
            urls = [
                "https://github.com/wniu9/dependencies/blob/master/gemmlowp.zip",
            ],
        )

def tflite_deps():
    if "tflite" not in native.existing_rules():
        http_archive(
            name = "tflite",
            sha256 = "1bb4571ee5cbde427ecfed076b39edaad96ace897ab86bb2495bdb93c706b203",
            strip_prefix = "tensorflow",
            urls = [
                "https://github.com/wniu9/dependencies/blob/master/tensorflow.zip",
            ],
        )

def arm_linux_deps():
    if "gcc_linaro_7_3_1_arm_linux_gnueabihf" not in native.existing_rules():
        http_archive(
            name = "gcc_linaro_7_3_1_arm_linux_gnueabihf",
            build_file = "//third_party:compilers/arm_compiler.BUILD",
            strip_prefix = "gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf",
            urls = [
                "https://releases.linaro.org/components/toolchain/binaries/7.3-2018.05/arm-linux-gnueabihf/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf.tar.xz",
            ],
        )

def aarch64_deps():
    if "gcc_linaro_7_3_1_aarch64_linux_gnu" not in native.existing_rules():
        http_archive(
            name = "gcc_linaro_7_3_1_aarch64_linux_gnu",
            build_file = "//third_party:compilers/aarch64_compiler.BUILD",
            strip_prefix = "gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu",
            sha256 = "73eed74e593e2267504efbcf3678918bb22409ab7afa3dc7c135d2c6790c2345",
            urls = [
                "https://releases.linaro.org/components/toolchain/binaries/7.3-2018.05/aarch64-linux-gnu/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz",
            ],
        )

def opencl_hdr_deps():
    if "opencl_headers_dep" not in native.existing_rules():
        http_archive(
            name = "opencl_headers_dep",
            build_file = "//third_party:opencl-headers/opencl-headers.BUILD",
            strip_prefix = "OpenCL-Headers-f039db6764d52388658ef15c30b2237bbda49803",
            sha256 = "b2b813dd88a7c39eb396afc153070f8f262504a7f956505b2049e223cfc2229b",
            urls = [
                "https://github.com/KhronosGroup/OpenCL-Headers/archive/f039db6764d52388658ef15c30b2237bbda49803.zip",
            ],
        )

def opencl_cpp_deps():
    if "opencl_clhpp" not in native.existing_rules():
        http_archive(
            name = "opencl_clhpp",
            build_file = "//third_party:opencl-clhpp/opencl-clhpp.BUILD",
            strip_prefix = "OpenCL-CLHPP-4c6f7d56271727e37fb19a9b47649dd175df2b12",
            sha256 = "dab6f1834ec6e3843438cc0f97d63817902aadd04566418c1fcc7fb78987d4e7",
            urls = [
                "https://github.com/KhronosGroup/OpenCL-CLHPP/archive/4c6f7d56271727e37fb19a9b47649dd175df2b12.zip",
            ],
        )

def gtest_deps():
    if "com_google_googletest" not in native.existing_rules():
        http_archive(
            name = "com_google_googletest",
            urls = [
                "https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip"
            ],
            strip_prefix = "googletest-release-1.12.1",
        )

    if "com_github_gflags_gflags" not in native.existing_rules():
        http_archive(
            name = "com_github_gflags_gflags",
            strip_prefix = "gflags-30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e",
            sha256 = "16903f6bb63c00689eee3bf7fb4b8f242934f6c839ce3afc5690f71b712187f9",
            urls = [
                "https://github.com/gflags/gflags/archive/30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e.zip",
            ],
        )

    native.bind(
        name = "gflags",
        actual = "@com_github_gflags_gflags//:gflags",
    )

    native.bind(
        name = "gflags_nothreads",
        actual = "@com_github_gflags_gflags//:gflags_nothreads",
    )

def half_deps(root):
    if "half" not in native.existing_rules():
        native.new_local_repository(
            name = "half",
            build_file = "//third_party:half/half.BUILD",
            path = "{}/third_party/half".format(root)
        )

def tensorflow_deps():
    if "org_tensorflow" in native.existing_rules():
        return
    http_archive(
        name = "org_tensorflow",
        strip_prefix = "tensorflow-2.9.1",
        sha256 = "9f2dac244e5af6c6a13a7dad6481e390174ac989931942098e7a4373f1bccfc2",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip",
        ],
        patches = [
        ],
        patch_args = [ "-p1" ],
    )

