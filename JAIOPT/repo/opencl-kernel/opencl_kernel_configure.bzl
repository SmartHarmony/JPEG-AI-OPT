"""Repository rule for opencl encrypt kernel autoconfiguration, borrow from tensorflow
"""

def _opencl_encrypt_kernel_impl(repository_ctx):
    repository_ctx.template(
        "BUILD",
        Label("//repo/opencl-kernel:BUILD.tpl"),
    )

    deepvan_root_path = str(repository_ctx.path(Label("@deepvan//:BUILD")))[:-len("BUILD")]
    generated_files_path = repository_ctx.path("gen")

    ret = repository_ctx.execute(
        ["test", "-f", "%s/.git/logs/HEAD" % deepvan_root_path],
    )
    if ret.return_code == 0:
        unused_var = repository_ctx.path(Label("//:.git/HEAD"))
    ret = repository_ctx.execute(
        ["test", "-f", "%s/.git/refs/heads/master" % deepvan_root_path],
    )
    if ret.return_code == 0:
        unused_var = repository_ctx.path(Label("//:.git/refs/heads/master"))

    repository_ctx.execute(
        ["test", "-f", "%s/deepvan/backend/opencl/cl/common.h" % deepvan_root_path],
    )

    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/activation.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/conv_2d.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/conv_2d_1x1.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/conv_2d_3x3.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/deconv_2d.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/eltwise.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/buffer_transform.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/winograd_transform.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/buffer_to_image.cl"))
    unused_var = repository_ctx.path(Label("//:deepvan/backend/opencl/cl/matmul.cl"))
    
    python_bin_path = repository_ctx.which("python")
    # python_bin_path = "/usr/local/bin/python3.7"

    repository_ctx.execute([
        python_bin_path,
        "%s/lothar/tools/encrypt_opencl_codegen.py" % deepvan_root_path,
        "--cl_kernel_dir=%s/deepvan/backend/opencl/cl" % deepvan_root_path,
        "--output_path=%s/encrypt_opencl_kernel" % generated_files_path,
    ], quiet = False)

encrypt_opencl_kernel_repository = repository_rule(
    implementation = _opencl_encrypt_kernel_impl,
)
