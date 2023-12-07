import glob
import logging
import os
import random
import re
import sh
import sys
import platform

import six
import hashlib
import json

from lothar.commons import common
from lothar.commons.common import abi_to_internal

try:
    from lothar.generate_data import generate_input_data
    from lothar.validate import validate
except Exception as e:
    six.print_("Import error:\n%s" % e, file=sys.stderr)
    exit(1)


################################
# common
################################

def strip_invalid_utf8(str):
    return sh.iconv(str, "-c", "-t", "UTF-8")


def split_stdout(stdout_str):
    stdout_str = strip_invalid_utf8(stdout_str)
    # Filter out last empty line
    return [l.strip() for l in stdout_str.split('\n') if len(l.strip()) > 0]


def make_output_processor(buff):
    def process_output(line):
        six.print_(line.rstrip())
        buff.append(line)

    return process_output


def device_lock_path(serialno):
    return "/tmp/device-lock-%s" % serialno


def device_lock(serialno, timeout=7200):
    import filelock
    return filelock.FileLock(device_lock_path(serialno), timeout=timeout)


class BuildType(object):
    proto = 'proto'
    code = 'code'


def stdout_success(stdout):
    stdout_lines = stdout.split("\n")
    for line in stdout_lines:
        if "Aborted" in line or "FAILED" in line or "Segmentation fault" in line:
            return False
    return True


################################
# clear data
################################
def clear_phone_data_dir(serialno, phone_data_dir):
    sh.adb("-s",
           serialno,
           "shell",
           "rm -rf %s" % phone_data_dir)


################################
# adb commands
################################
def adb_devices():
    serialnos = []
    p = re.compile(r'(\S+)\s+device')
    for line in split_stdout(sh.adb("devices")):
        m = p.match(line)
        if m:
            serialnos.append(m.group(1))

    return serialnos


def adb_getprop_by_serialno(serialno):
    outputs = sh.adb("-s", serialno, "shell", "getprop")
    raw_props = split_stdout(outputs)
    props = {}
    p = re.compile(r'\[(.+)\]: \[(.+)\]')
    for raw_prop in raw_props:
        m = p.match(raw_prop)
        if m:
            props[m.group(1)] = m.group(2)
    return props


def adb_get_all_socs():
    socs = []
    for d in adb_devices():
        props = adb_getprop_by_serialno(d)
        socs.append(props["ro.board.platform"])
    return set(socs)


def adb_push(src_path, dst_path, serialno):
    sh.adb("-s", serialno, "push", src_path, dst_path)


def adb_pull(src_path, dst_path, serialno):
    try:
        sh.adb("-s", serialno, "pull", src_path, dst_path)
    except Exception as e:
        six.print_("Error msg: %s" % e, file=sys.stderr)


def compare_md5(local_file, remote_file, serialno):
    try:
        result = sh.adb("-s", serialno, "shell", "md5sum", remote_file)
        remote_md5 = result.split(" ")[0]
        local_md5 = check_md5(local_file)
        return remote_md5 == local_md5
    except Exception:
        print("Compared model data's md5 failed")
        return False


def delete_files(start_with, data_dir, serialno):
    try:
        result = sh.adb("-s", serialno, "shell", "ls", data_dir)
        print(result)
        for file in result:
            if any(file.startswith(s) for s in start_with):
                sh.adb("-s", serialno, "shell", "rm -f",
                       "%s/%s" % (data_dir, file))
    except Exception:
        print("Delete file start with {} failed.".format(start_with))


################################
# Toolchain
################################


################################
# bazel commands
################################
def bazel_build(target,
                abi="armeabi-v7a",
                toolchain='android',
                enable_hexagon=False,
                enable_hta=False,
                enable_openmp=True,
                enable_neon=True,
                enable_opencl=True,
                enable_quantize=True,
                enable_fallback=False,
                address_sanitizer=False,
                symbol_hidden=True,
                debug_enable=False,
                static_openmp=False,
                extra_args=""):
    six.print_("* Build %s with ABI %s" % (target, abi))
    if abi == "host":
        toolchain = platform.system().lower()
        bazel_args = (
            "build",
            "--config",
            toolchain,
            "--define",
            "openmp=%s" % str(enable_openmp).lower(),
            "--define",
            "quantize=%s" % str(enable_quantize).lower(),
            target,
        )
    else:
        bazel_args = (
            "build",
            target,
            "--config",
            toolchain,
            "--cpu=%s" % abi_to_internal(abi),
            "--define",
            "neon=%s" % str(enable_neon).lower(),
            "--define",
            "openmp=%s" % str(enable_openmp).lower(),
            "--define",
            "static_openmp=%s" % str(static_openmp).lower(),
            "--define",
            "opencl=%s" % str(enable_opencl).lower(),
            "--define",
            "quantize=%s" % str(enable_quantize).lower(),
            "--define",
            "hexagon=%s" % str(enable_hexagon).lower(),
            "--define",
            "hta=%s" % str(enable_hta).lower(),
            "--define",
            "fallback=%s" % str(enable_fallback).lower())

    if address_sanitizer:
        bazel_args += ("--config", "asan")
    if debug_enable:
        bazel_args += ("--config", "debug")
    if not address_sanitizer and not debug_enable:
        if toolchain == "darwin" or toolchain == "ios":
            bazel_args += ("--config", "optimization_darwin")
        else:
            bazel_args += ("--config", "optimization")
        if symbol_hidden:
            bazel_args += ("--config", "symbol_hidden")
    if enable_fallback:
        bazel_args += (
            "--fat_apk_cpu=arm64_v8a",
            "--define", "tflite_with_xnnpack=true",
            "--define", "xnn_enable_qs8=true",
            "--define", "xnn_enable_qu8=true")
    if extra_args:
        bazel_args += (extra_args,)
        six.print_(bazel_args)
    print(bazel_args)
    print('Current Working Dir: ', os.getcwd())
    sh.bazel(
        _fg=True,
        *bazel_args)
    six.print_("Build done!\n")


def bazel_build_common(target, build_args=""):
    stdout_buff = []
    process_output = make_output_processor(stdout_buff)
    print("bazel build " + target + build_args)
    sh.bazel(
        "build",
        target + build_args,
        _tty_in=True,
        _out=process_output,
        _err_to_out=True)
    return "".join(stdout_buff)


################################
# deepvan commands
################################
def gen_model_code(model_codegen_dir,
                   platform,
                   model_file_path,
                   weight_file_path,
                   input_nodes,
                   input_data_types,
                   input_data_formats,
                   output_nodes,
                   output_data_types,
                   output_data_formats,
                   check_nodes,
                   runtime,
                   model_tag,
                   input_shapes,
                   input_ranges,
                   output_shapes,
                   check_shapes,
                   dsp_mode,
                   winograd,
                   quantize,
                   quantize_range_file,
                   pattern_weight,
                   pattern_config_path,
                   pattern_style_count,
                   conv_unroll,
                   change_concat_ranges,
                   obfuscate,
                   data_type,
                   model_type,
                   cl_mem_type,
                   graph_optimize_options,
                   load_from_im=False,
                   pruning_type='',
                   executing_devices=None,
                   nuitka=False):
    if executing_devices is None:
        executing_devices = {}

    if os.path.exists(model_codegen_dir):
        sh.rm("-rf", model_codegen_dir)
    sh.mkdir("-p", model_codegen_dir)

    if not nuitka:
        bazel_build_common("//lothar:converter")

        sh.python3("bazel-bin/lothar/converter",
                   "-u",
                   "--platform=%s" % platform,
                   "--model_file=%s" % model_file_path,
                   "--weight_file=%s" % weight_file_path,
                   "--input_node=%s" % input_nodes,
                   "--input_data_types=%s" % input_data_types,
                   "--input_data_formats=%s" % input_data_formats,
                   "--output_node=%s" % output_nodes,
                   "--output_data_types=%s" % output_data_types,
                   "--output_data_formats=%s" % output_data_formats,
                   "--check_node=%s" % check_nodes,
                   "--runtime=%s" % runtime,
                   "--template=%s" % "lothar/tools",
                   "--model_tag=%s" % model_tag,
                   "--input_shape=%s" % input_shapes,
                   "--input_range=%s" % input_ranges,
                   "--output_shape=%s" % output_shapes,
                   "--check_shape=%s" % check_shapes,
                   "--dsp_mode=%s" % dsp_mode,
                   "--winograd=%s" % winograd,
                   "--quantize=%s" % quantize,
                   "--quantize_range_file=%s" % quantize_range_file,
                   "--pattern_weight=%s" % pattern_weight,
                   "--pattern_config_path=%s" % pattern_config_path,
                   "--pattern_style_count=%s" % pattern_style_count,
                   "--conv_unroll=%s" % conv_unroll,
                   "--change_concat_ranges=%s" % change_concat_ranges,
                   "--obfuscate=%s" % obfuscate,
                   "--output_dir=%s" % model_codegen_dir,
                   "--data_type=%s" % data_type,
                   "--model_type=%s" % model_type,
                   "--graph_optimize_options=%s" % graph_optimize_options,
                   "--cl_mem_type=%s" % cl_mem_type,
                   "--load_from_im=%s" % load_from_im,
                   "--pruning_type=%s" % pruning_type,
                   "--executing_devices=%s" % json.dumps(executing_devices),
                   _fg=True)
    else:
        nuitka_dir = "nuitka3-bin/lothar"
        converter_dist = os.path.join(nuitka_dir, "converter.dist")
        converter_bin_path = os.path.join(converter_dist, 'converter')
        # need manually rm -rf nuitka3-bin/lothar/converter.dist, when converter python code changed.
        if not os.path.exists(converter_bin_path):
            bazel_build_common("//lothar:converter")
            sh.cp("-f", "bazel-bin/deepvan/proto/deepvan_pb2.py", "deepvan/proto/")
            sh.cp("-f", "bazel-bin/third_party/caffe/caffe_pb2.py",
                  "third_party/caffe/")

            sh.nuitka3("--standalone",
                       "--show-progress",
                       "--show-memory",
                       "--nofollow-import-to=tensorflow,matplotlib,multiprocessing,numpy,onnx,jinja2,markupsafe,google.protobuf",
                       "--output-dir=%s" % nuitka_dir,
                       "lothar/tools/converter.py",
                       _fg=True)

            # copy dependece packages
            for package in ("numpy", "numpy.libs", "onnx", "jinja2", "markupsafe", "google/protobuf"):
                for path in sys.path:
                    package_path = os.path.join(path, package)
                    if os.path.exists(package_path):
                        if package.startswith('google/'):
                            converter_dist_google = os.path.join(
                                converter_dist, 'google')
                            sh.mkdir("-p", converter_dist_google)
                            sh.cp("-rf", package_path, converter_dist_google)
                        else:
                            sh.cp("-rf", package_path, converter_dist)
            import cmath
            sh.cp("-rf", cmath.__file__, converter_dist)
        converter = sh.Command(converter_bin_path)
        converter("-u",
                  "--platform=%s" % platform,
                  "--model_file=%s" % model_file_path,
                  "--weight_file=%s" % weight_file_path,
                  "--input_node=%s" % input_nodes,
                  "--input_data_types=%s" % input_data_types,
                  "--input_data_formats=%s" % input_data_formats,
                  "--output_node=%s" % output_nodes,
                  "--output_data_types=%s" % output_data_types,
                  "--output_data_formats=%s" % output_data_formats,
                  "--check_node=%s" % check_nodes,
                  "--runtime=%s" % runtime,
                  "--template=%s" % "lothar/tools",
                  "--model_tag=%s" % model_tag,
                  "--input_shape=%s" % input_shapes,
                  "--input_range=%s" % input_ranges,
                  "--output_shape=%s" % output_shapes,
                  "--check_shape=%s" % check_shapes,
                  "--dsp_mode=%s" % dsp_mode,
                  "--winograd=%s" % winograd,
                  "--quantize=%s" % quantize,
                  "--quantize_range_file=%s" % quantize_range_file,
                  "--pattern_weight=%s" % pattern_weight,
                  "--pattern_config_path=%s" % pattern_config_path,
                  "--pattern_style_count=%s" % pattern_style_count,
                  "--conv_unroll=%s" % conv_unroll,
                  "--change_concat_ranges=%s" % change_concat_ranges,
                  "--obfuscate=%s" % obfuscate,
                  "--output_dir=%s" % model_codegen_dir,
                  "--data_type=%s" % data_type,
                  "--model_type=%s" % model_type,
                  "--graph_optimize_options=%s" % graph_optimize_options,
                  "--cl_mem_type=%s" % cl_mem_type,
                  "--load_from_im=%s" % load_from_im,
                  "--pruning_type=%s" % pruning_type,
                  "--executing_devices=%s" % json.dumps(executing_devices),
                  _fg=True)


def gen_random_input(model_output_dir,
                     input_nodes,
                     input_shapes,
                     input_files,
                     input_ranges,
                     input_data_types,
                     input_file_name="model_input"):
    for input_name in input_nodes:
        formatted_name = common.formatted_file_name(
            input_file_name, input_name)
        if os.path.exists("%s/%s" % (model_output_dir, formatted_name)):
            sh.rm("%s/%s" % (model_output_dir, formatted_name))
    input_nodes_str = ",".join(input_nodes)
    input_shapes_str = ":".join(input_shapes)
    input_ranges_str = ":".join(input_ranges)
    input_data_types_str = ",".join(input_data_types)
    generate_input_data("%s/%s" % (model_output_dir, input_file_name),
                        input_nodes_str,
                        input_shapes_str,
                        input_ranges_str,
                        input_data_types_str)

    input_file_list = []
    if isinstance(input_files, list):
        input_file_list.extend(input_files)
    else:
        input_file_list.append(input_files)
    if len(input_file_list) != 0:
        input_name_list = []
        if isinstance(input_nodes, list):
            input_name_list.extend(input_nodes)
        else:
            input_name_list.append(input_nodes)
        if len(input_file_list) != len(input_name_list):
            raise Exception('If input_files set, the input files should '
                            'match the input names.')
        for i in range(len(input_file_list)):
            if input_file_list[i] is not None:
                dst_input_file = model_output_dir + '/' + \
                    common.formatted_file_name(input_file_name,
                                               input_name_list[i])
                if input_file_list[i].startswith("http://") or \
                        input_file_list[i].startswith("https://"):
                    six.moves.urllib.request.urlretrieve(input_file_list[i],
                                                         dst_input_file)
                else:
                    sh.cp("-f", input_file_list[i], dst_input_file)


def update_deepvan_run_binary(build_tmp_binary_dir, link_dynamic=False):
    if link_dynamic:
        deepvan_filepath = build_tmp_binary_dir + "/deepvan_run_dynamic"
    else:
        deepvan_filepath = build_tmp_binary_dir + "/deepvan_run_static"

    if os.path.exists(deepvan_filepath):
        sh.rm("-rf", deepvan_filepath)
    if link_dynamic:
        sh.cp("-f", "bazel-bin/deepvan/run/deepvan_run_dynamic",
              build_tmp_binary_dir)
    else:
        sh.cp("-f", "bazel-bin/deepvan/run/deepvan_run_static",
              build_tmp_binary_dir)


def create_internal_storage_dir(serialno, phone_data_dir):
    internal_storage_dir = "%s/interior/" % phone_data_dir
    sh.adb("-s", serialno, "shell", "mkdir", "-p", internal_storage_dir)
    return internal_storage_dir

def validate_model(abi,
                   device,
                   model_file_path,
                   weight_file_path,
                   platform,
                   device_type,
                   input_nodes,
                   output_nodes,
                   input_shapes,
                   output_shapes,
                   input_data_formats,
                   output_data_formats,
                   model_output_dir,
                   input_data_types,
                   input_file_name="model_input",
                   output_file_name="model_out",
                   validation_threshold=0.9,
                   backend="tensorflow",
                   validation_outputs_data=[],
                   log_file=""):
    if not validation_outputs_data:
        six.print_("* Validate with %s" % platform)
    else:
        six.print_("* Validate with file: %s" % validation_outputs_data)
    if abi != "host":
        for output_name in output_nodes:
            formatted_name = common.formatted_file_name(
                output_file_name, output_name)
            if os.path.exists("%s/%s" % (model_output_dir,
                                         formatted_name)):
                sh.rm("-rf", "%s/%s" % (model_output_dir, formatted_name))
            device.pull_from_data_dir(formatted_name, model_output_dir)

    if platform == "tensorflow" or platform == "onnx":
        validate(platform, model_file_path, "",
                 "%s/%s" % (model_output_dir, input_file_name),
                 "%s/%s" % (model_output_dir, output_file_name), device_type,
                 ":".join(input_shapes), ":".join(output_shapes),
                 ",".join(input_data_formats), ",".join(output_data_formats),
                 ",".join(input_nodes), ",".join(output_nodes),
                 validation_threshold, ",".join(input_data_types), backend,
                 validation_outputs_data,
                 log_file)
    six.print_("Validation done!\n")

################################
# benchmark
################################
def build_run_throughput_test(abi,
                              serialno,
                              vlog_level,
                              run_seconds,
                              merged_lib_file,
                              model_input_dir,
                              input_nodes,
                              output_nodes,
                              input_shapes,
                              output_shapes,
                              cpu_model_tag,
                              gpu_model_tag,
                              dsp_model_tag,
                              phone_data_dir,
                              strip="always",
                              input_file_name="model_input"):
    six.print_("* Build and run throughput_test")

    model_tag_build_flag = ""
    if cpu_model_tag:
        model_tag_build_flag += "--copt=-DCPU_MODEL_TAG=%s " % \
                                cpu_model_tag
    if gpu_model_tag:
        model_tag_build_flag += "--copt=-DGPU_MODEL_TAG=%s " % \
                                gpu_model_tag
    if dsp_model_tag:
        model_tag_build_flag += "--copt=-DDSP_MODEL_TAG=%s " % \
                                dsp_model_tag

    sh.cp("-f", merged_lib_file, "deepvan/run/executor_merged.a")
    sh.bazel(
        "build",
        "-c",
        "opt",
        "--strip",
        strip,
        "--verbose_failures",
        "//deepvan/run:model_throughput_test",
        "--crosstool_top=//external:android/crosstool",
        "--host_crosstool_top=@bazel_tools//tools/cpp:toolchain",
        "--cpu=%s" % abi,
        "--copt=-std=c++11",
        "--copt=-D_GLIBCXX_USE_C99_MATH_TR1",
        "--copt=-Werror=return-type",
        "--copt=-O3",
        "--define",
        "neon=true",
        "--define",
        "openmp=true",
        model_tag_build_flag,
        _fg=True)

    sh.rm("deepvan/run/executor_merged.a")
    sh.adb("-s",
           serialno,
           "shell",
           "mkdir",
           "-p",
           phone_data_dir)
    adb_push("%s/%s_%s" % (model_input_dir, input_file_name,
                           ",".join(input_nodes)),
             phone_data_dir,
             serialno)
    adb_push("bazel-bin/deepvan/run/model_throughput_test",
             phone_data_dir,
             serialno)
    adb_push("codegen/models/%s/%s.data" % cpu_model_tag,
             phone_data_dir,
             serialno)
    adb_push("codegen/models/%s/%s.data" % gpu_model_tag,
             phone_data_dir,
             serialno)
    adb_push("codegen/models/%s/%s.data" % dsp_model_tag,
             phone_data_dir,
             serialno)

    adb_push("third_party/nnlib/%s/libhexagon_controller.so" % abi,
             phone_data_dir,
             serialno)

    sh.adb(
        "-s",
        serialno,
        "shell",
        "LD_LIBRARY_PATH=%s" % phone_data_dir,
        "DEEPVAN_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
        "DEEPVAN_RUN_PARAMETER_PATH=%s/deepvan_run.config" %
        phone_data_dir,
        "%s/model_throughput_test" % phone_data_dir,
        "--input_node=%s" % ",".join(input_nodes),
        "--output_node=%s" % ",".join(output_nodes),
        "--input_shape=%s" % ":".join(input_shapes),
        "--output_shape=%s" % ":".join(output_shapes),
        "--input_file=%s/%s" % (phone_data_dir, input_file_name),
        "--cpu_model_data_file=%s/%s.data" % (phone_data_dir,
                                              cpu_model_tag),
        "--gpu_model_data_file=%s/%s.data" % (phone_data_dir,
                                              gpu_model_tag),
        "--dsp_model_data_file=%s/%s.data" % (phone_data_dir,
                                              dsp_model_tag),
        "--run_seconds=%s" % run_seconds,
        _fg=True)

    six.print_("throughput_test done!\n")
