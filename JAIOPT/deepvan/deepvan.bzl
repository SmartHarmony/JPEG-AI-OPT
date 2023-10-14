# -*- Python -*-

def is_android(a, default_value = []):
  return select({
      "//deepvan:android": a,
      "//conditions:default": default_value,
  })

def is_ios(a, default_value = []):
  return select({
      "//deepvan:ios": a,
      "//conditions:default": default_value,
  })

def is_linux(a, default_value = []):
  return select({
      "//deepvan:linux": a,
      "//conditions:default": default_value,
  })

def is_darwin(a, default_value = []):
  return select({
      "//deepvan:darwin": a,
      "//conditions:default": default_value,
  })

def is_android_armv7(a):
  return select({
      "//deepvan:android_armv7": a,
      "//conditions:default": [],
  })

def is_android_arm64(a):
  return select({
      "//deepvan:android_arm64": a,
      "//conditions:default": [],
  })

def is_arm_linux_aarch64(a):
  return select({
      "//deepvan:arm_linux_aarch64": a,
      "//conditions:default": [],
  })

def is_arm_linux_armhf(a):
  return select({
      "//deepvan:arm_linux_armhf": a,
      "//conditions:default": []
  })

def is_neon_support(a, default_value = []):
  return select({
      "//deepvan:neon_support": a,
      "//conditions:default": default_value,
  })

def is_hexagon_support(a):
  return select({
      "//deepvan:hexagon_support": a,
      "//conditions:default": [],
  })

def is_not_hexagon_support(a):
  return select({
      "//deepvan:hexagon_support": [],
      "//conditions:default": a,
  })

def is_hta_support(a):
  return select({
      "//deepvan:hta_support": a,
      "//conditions:default": [],
  })

def is_hexagon_or_hta_support(a):
  return select({
      "//deepvan:hexagon_support": a,
      "//deepvan:hta_support": a,
      "//conditions:default": [],
  })

def is_openmp_support(a):
  return select({
      "//deepvan:openmp_support": a,
      "//conditions:default": [],
  })

def is_static_openmp_support(a):
  return select({
      "//deepvan:static_openmp_support": a,
      "//conditions:default": [],
  })

def is_opencl_support(a, default_value = []):
  return select({
      "//deepvan:opencl_support": a,
      "//conditions:default": default_value,
  })


def encrypt_opencl_kernel_genrule():
  native.genrule(
      name = "encrypt_opencl_kernel_gen",
      srcs = [str(Label("@local_opencl_kernel_encrypt//:gen/encrypt_opencl_kernel"))],
      outs = ["opencl/encrypt_opencl_kernel.cc"],
      cmd = "cat $(SRCS) > $@;"
  )

def is_memprof_support(a):
  return select({
      "//deepvan:memprof_support": a,
      "//conditions:default": [],
  })

def is_fallback_support(a):
  return select({
      "//deepvan:fallback_support": a,
      "//conditions:default": [],
  })
