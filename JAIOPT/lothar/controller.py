import argparse
import re
import sh
import sys
import copy
import time
import yaml

from enum import Enum
import six
import os

try:
    dirname = "/".join(os.path.dirname(os.path.abspath(__file__)
                                       ).split("/")[0:-1])
    sys.path.insert(0, dirname)
except Exception as e:
    print("Change work directory failed")
    exit(1)
from lothar.commons import command_helper
from lothar.commons.common import *
from lothar.commons.running_helper import DeviceWrapper, DeviceManager

from lothar.commons import common


################################
# set environment
################################
os.environ['CPP_MIN_LOG_LEVEL'] = '2'

################################
# common definitions
################################

ABITypeStrs = [
    'armeabi-v7a',
    'arm64-v8a',
    'arm64',
    'armhf',
    'host',
]

PlatformTypeStrs = [
    "tensorflow",
    "caffe",
    "onnx",
]
PlatformType = Enum('PlatformType', [(ele, ele) for ele in PlatformTypeStrs],
                    type=str)

RuntimeTypeStrs = [
    "cpu",
    "gpu",
    "dsp",
    "hta",
    "cpu+gpu"
]

InOutDataTypeStrs = [
    "int8",
    "uint8",
    "int32",
    "float32",
]

InOutDataType = Enum('InputDataType',
                     [(ele, ele) for ele in InOutDataTypeStrs],
                     type=str)

FPDataTypeStrs = [
    "float16_float32",
    "float32_float32",
]

FPDataType = Enum('GPUDataType', [(ele, ele) for ele in FPDataTypeStrs],
                  type=str)

DSPDataTypeStrs = [
    "uint8",
]

DSPDataType = Enum('DSPDataType', [(ele, ele) for ele in DSPDataTypeStrs],
                   type=str)

WinogradParameters = [0, 2, 4]

DataFormatStrs = [
    "NONE",
    "NHWC",
    "NCHW",
    "OIHW",
]

PruningTypeStrs = [
    "DENSE",
]


class DefaultValues(object):
    deepvan_lib_type = DEEPVANLibType.static
    omp_num_threads = -1,
    cpu_affinity_policy = -1,
    gpu_perf_hint = 3,
    gpu_priority_hint = 3,


class ValidationThreshold(object):
    cpu_threshold = 0.999,
    gpu_threshold = 0.995,
    hexagon_threshold = 0.930,
    cpu_quantize_threshold = 0.980,


################################
# common functions
################################
def parse_device_type(runtime):
    device_type = ""

    if runtime == RuntimeType.dsp:
        device_type = DeviceType.HEXAGON
    elif runtime == RuntimeType.hta:
        device_type = DeviceType.HTA
    elif runtime == RuntimeType.gpu:
        device_type = DeviceType.GPU
    elif runtime == RuntimeType.cpu:
        device_type = DeviceType.CPU

    return device_type


def read_config_fields(configs, key, default=""):
    result = []
    for model_name in configs[YAMLKeyword.models]:
        value = \
            configs[YAMLKeyword.models][model_name].get(
                key, default)
        result.append(value.lower())
    return result


def read_config_field(configs, key, default=""):
    for model_name in configs[YAMLKeyword.models]:
        value = \
            configs[YAMLKeyword.models][model_name].get(
                key, default)
        if value != default:
            return value
    return default


def get_hexagon_mode(configs):
    runtime_list = read_config_fields(configs, YAMLKeyword.runtime)
    return RuntimeType.dsp in runtime_list


def get_hta_mode(configs):
    runtime_list = read_config_fields(configs, YAMLKeyword.runtime)
    return RuntimeType.hta in runtime_list


def get_opencl_mode(configs):
    # TODO @vgod
    runtime_list = read_config_fields(configs, YAMLKeyword.runtime)
    return True
    # return RuntimeType.gpu in runtime_list or RuntimeType.cpu_gpu in runtime_list


def get_quantize_mode(configs):
    return read_config_field(configs, YAMLKeyword.quantize, 0) != 0


def get_pattern_weight(configs):
    return read_config_field(configs, YAMLKeyword.pattern_weight, 0)


def get_pattern_config_path(configs):
    return read_config_field(configs, YAMLKeyword.pattern_config_path, '')


def get_runtime_list(configs):
    return read_config_fields(configs, YAMLKeyword.runtime)


def get_pruning_type(configs):
    return read_config_field(configs, YAMLKeyword.pruning_type).upper()


def get_conv_unroll(configs):
    return read_config_field(configs, YAMLKeyword.conv_unroll, 1)


def get_pattern_style_count(configs):
    return read_config_field(configs, YAMLKeyword.pattern_style_count, 0)


def is_load_from_im(configs):
    return read_config_field(configs, YAMLKeyword.load_from_im, False)


def get_executing_devices(configs):
    return read_config_field(configs, YAMLKeyword.executing_devices, {})


def get_symbol_hidden_mode(debug_enable, deepvan_lib_type=None):
    if not deepvan_lib_type:
        return True
    if debug_enable or deepvan_lib_type == DEEPVANLibType.dynamic:
        return False
    else:
        return True


def md5sum(str):
    md5 = hashlib.md5()
    md5.update(str.encode('utf-8'))
    return md5.hexdigest()


def sha256_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def format_model_config(flags):
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper
    with open(flags.config) as f:
        configs = yaml.load(f, Loader)

    library_name = configs.get(YAMLKeyword.library_name, "")
    CONDITIONS(len(library_name) > 0,
               ModuleName.YAML_CONFIG, "library name should not be empty")

    target_abis = configs.get(YAMLKeyword.target_abis, [])
    CONDITIONS((isinstance(target_abis, list) and len(target_abis) > 0),
               ModuleName.YAML_CONFIG, "target_abis list is needed")
    configs[YAMLKeyword.target_abis] = target_abis
    for abi in target_abis:
        CONDITIONS(abi in ABITypeStrs,
                   ModuleName.YAML_CONFIG,
                   "target_abis must be in " + str(ABITypeStrs))

    target_socs = configs.get(YAMLKeyword.target_socs, "")
    if flags.target_socs and flags.target_socs != TargetSOCTag.random \
            and flags.target_socs != TargetSOCTag.all:
        configs[YAMLKeyword.target_socs] = \
            [soc.lower() for soc in flags.target_socs.split(',')]
    elif not target_socs:
        configs[YAMLKeyword.target_socs] = []
    elif not isinstance(target_socs, list):
        configs[YAMLKeyword.target_socs] = [target_socs]

    configs[YAMLKeyword.target_socs] = \
        [soc.lower() for soc in configs[YAMLKeyword.target_socs]]

    if ABIType.armeabi_v7a in target_abis \
            or ABIType.arm64_v8a in target_abis:
        available_socs = command_helper.adb_get_all_socs()
        target_socs = configs[YAMLKeyword.target_socs]
        if TargetSOCTag.all in target_socs:
            CONDITIONS(available_socs,
                       ModuleName.YAML_CONFIG,
                       "Android abi is listed in config file and "
                       "build for all SOCs plugged in computer, "
                       "But no android phone found, "
                       "you at least plug in one phone")
        else:
            for soc in target_socs:
                CONDITIONS(soc in available_socs,
                           ModuleName.YAML_CONFIG,
                           "Build specified SOC library, "
                           "you must plug in a phone using the SOC")

    model_names = configs.get(YAMLKeyword.models, [])
    CONDITIONS(len(model_names) > 0, ModuleName.YAML_CONFIG,
               "no model found in config file")

    model_name_reg = re.compile(r'^[a-zA-Z0-9_]+$')
    for model_name in model_names:
        # check model_name legality
        CONDITIONS((model_name[0] == '_' or model_name[0].isalpha())
                   and bool(model_name_reg.match(model_name)),
                   ModuleName.YAML_CONFIG,
                   "model name should Meet the c++ naming convention"
                   " which start with '_' or alpha"
                   " and only contain alpha, number and '_'")

        model_config = configs[YAMLKeyword.models][model_name]
        platform = model_config.get(YAMLKeyword.platform, "")
        CONDITIONS(platform in PlatformTypeStrs,
                   ModuleName.YAML_CONFIG,
                   "'platform' must be in " + str(PlatformTypeStrs))

        for key in [YAMLKeyword.model_file_path]:
            value = model_config.get(key, "")
            if len(flags.model_path) > 0:
                model_config[key] = flags.model_path
                with open(flags.config, "w") as f:
                    yaml.dump(configs, f)
                print(f"Save yaml to {flags.config}")
                flags.model_path = ""
            CONDITIONS(value != "", ModuleName.YAML_CONFIG,
                       "'%s' is necessary" % key)

        weight_file_path = model_config.get(YAMLKeyword.weight_file_path, "")

        runtime = model_config.get(YAMLKeyword.runtime, "")
        CONDITIONS(runtime in RuntimeTypeStrs,
                   ModuleName.YAML_CONFIG,
                   "'runtime' must be in " + str(RuntimeTypeStrs))
        if ABIType.host in target_abis:
            CONDITIONS(runtime == RuntimeType.cpu,
                       ModuleName.YAML_CONFIG,
                       "host only support cpu runtime now.")

        data_type = model_config.get(YAMLKeyword.data_type, "")
        if runtime == RuntimeType.dsp:
            if len(data_type) > 0:
                CONDITIONS(data_type in DSPDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(DSPDataTypeStrs)
                           + " for dsp runtime")
            else:
                model_config[YAMLKeyword.data_type] = \
                    DSPDataType.uint8.value
        else:
            if len(data_type) > 0:
                CONDITIONS(data_type in FPDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(FPDataTypeStrs)
                           + " for cpu runtime")
            else:
                if runtime == RuntimeType.cpu:
                    model_config[YAMLKeyword.data_type] = \
                        FPDataType.float32_float32.value
                else:
                    model_config[YAMLKeyword.data_type] = \
                        FPDataType.float16_float32.value

        subgraphs = model_config.get(YAMLKeyword.subgraphs, "")
        CONDITIONS(len(subgraphs) > 0, ModuleName.YAML_CONFIG,
                   "at least one subgraph is needed")

        for subgraph in subgraphs:
            for key in [YAMLKeyword.input_tensors,
                        YAMLKeyword.input_shapes,
                        YAMLKeyword.output_tensors,
                        YAMLKeyword.output_shapes]:
                value = subgraph.get(key, "")
                CONDITIONS(value != "", ModuleName.YAML_CONFIG,
                           "'%s' is necessary in subgraph" % key)
                if not isinstance(value, list):
                    subgraph[key] = [value]
                subgraph[key] = [str(v) for v in subgraph[key]]
            input_size = len(subgraph[YAMLKeyword.input_tensors])
            output_size = len(subgraph[YAMLKeyword.output_tensors])

            CONDITIONS(len(subgraph[YAMLKeyword.input_shapes]) == input_size,
                       ModuleName.YAML_CONFIG,
                       "input shapes' size not equal inputs' size.")
            CONDITIONS(len(subgraph[YAMLKeyword.output_shapes]) == output_size,
                       ModuleName.YAML_CONFIG,
                       "output shapes' size not equal outputs' size.")

            for key in [YAMLKeyword.check_tensors,
                        YAMLKeyword.check_shapes]:
                value = subgraph.get(key, "")
                if value != "":
                    if not isinstance(value, list):
                        subgraph[key] = [value]
                    subgraph[key] = [str(v) for v in subgraph[key]]
                else:
                    subgraph[key] = []

            for key in [YAMLKeyword.input_data_types,
                        YAMLKeyword.output_data_types]:
                if key == YAMLKeyword.input_data_types:
                    count = input_size
                else:
                    count = output_size
                data_types = subgraph.get(key, "")
                if data_types:
                    if not isinstance(data_types, list):
                        subgraph[key] = [data_types] * count
                    for data_type in subgraph[key]:
                        CONDITIONS(data_type in InOutDataTypeStrs,
                                   ModuleName.YAML_CONFIG,
                                   key + " must be in "
                                   + str(InOutDataTypeStrs))
                else:
                    subgraph[key] = [InOutDataType.float32] * count

            input_data_formats = subgraph.get(YAMLKeyword.input_data_formats,
                                              [])
            if input_data_formats:
                if not isinstance(input_data_formats, list):
                    subgraph[YAMLKeyword.input_data_formats] = \
                        [input_data_formats] * input_size
                else:
                    CONDITIONS(len(input_data_formats)
                               == input_size,
                               ModuleName.YAML_CONFIG,
                               "input_data_formats should match"
                               " the size of input.")
                for input_data_format in \
                        subgraph[YAMLKeyword.input_data_formats]:
                    CONDITIONS(input_data_format in DataFormatStrs,
                               ModuleName.YAML_CONFIG,
                               "'input_data_formats' must be in "
                               + str(DataFormatStrs) + ", but got "
                               + input_data_format)
            else:
                subgraph[YAMLKeyword.input_data_formats] = \
                    [DataFormat.NHWC] * input_size

            output_data_formats = subgraph.get(YAMLKeyword.output_data_formats,
                                               [])
            if output_data_formats:
                if not isinstance(output_data_formats, list):
                    subgraph[YAMLKeyword.output_data_formats] = \
                        [output_data_formats] * output_size
                else:
                    CONDITIONS(len(output_data_formats)
                               == output_size,
                               ModuleName.YAML_CONFIG,
                               "output_data_formats should match"
                               " the size of output")
                for output_data_format in \
                        subgraph[YAMLKeyword.output_data_formats]:
                    CONDITIONS(output_data_format in DataFormatStrs,
                               ModuleName.YAML_CONFIG,
                               "'output_data_formats' must be in "
                               + str(DataFormatStrs))
            else:
                subgraph[YAMLKeyword.output_data_formats] = \
                    [DataFormat.NHWC] * output_size

            validation_threshold = subgraph.get(
                YAMLKeyword.validation_threshold, {})
            if not isinstance(validation_threshold, dict):
                raise argparse.ArgumentTypeError(
                    'similarity threshold must be a dict.')

            threshold_dict = {
                DeviceType.CPU: ValidationThreshold.cpu_threshold,
                DeviceType.GPU: ValidationThreshold.gpu_threshold,
                DeviceType.HEXAGON + "_QUANTIZE":
                    ValidationThreshold.hexagon_threshold,
                DeviceType.HTA + "_QUANTIZE":
                    ValidationThreshold.hexagon_threshold,
                DeviceType.CPU + "_QUANTIZE":
                    ValidationThreshold.cpu_quantize_threshold,
            }
            for k, v in six.iteritems(validation_threshold):
                if k.upper() == 'DSP':
                    k = DeviceType.HEXAGON
                if k.upper() not in (DeviceType.CPU,
                                     DeviceType.GPU,
                                     DeviceType.HEXAGON,
                                     DeviceType.HTA,
                                     DeviceType.CPU + "_QUANTIZE"):
                    raise argparse.ArgumentTypeError(
                        'Unsupported validation threshold runtime: %s' % k)
                threshold_dict[k.upper()] = v

            subgraph[YAMLKeyword.validation_threshold] = threshold_dict

            validation_inputs_data = subgraph.get(
                YAMLKeyword.validation_inputs_data, [])
            if not isinstance(validation_inputs_data, list):
                subgraph[YAMLKeyword.validation_inputs_data] = [
                    validation_inputs_data]
            else:
                subgraph[YAMLKeyword.validation_inputs_data] = \
                    validation_inputs_data

            onnx_backend = subgraph.get(
                YAMLKeyword.backend, "tensorflow")
            subgraph[YAMLKeyword.backend] = onnx_backend
            validation_outputs_data = subgraph.get(
                YAMLKeyword.validation_outputs_data, [])
            if not isinstance(validation_outputs_data, list):
                subgraph[YAMLKeyword.validation_outputs_data] = [
                    validation_outputs_data]
            else:
                subgraph[YAMLKeyword.validation_outputs_data] = \
                    validation_outputs_data
            input_ranges = subgraph.get(
                YAMLKeyword.input_ranges, [])
            if not isinstance(input_ranges, list):
                subgraph[YAMLKeyword.input_ranges] = [input_ranges]
            else:
                subgraph[YAMLKeyword.input_ranges] = input_ranges
            subgraph[YAMLKeyword.input_ranges] = \
                [str(v) for v in subgraph[YAMLKeyword.input_ranges]]

        for key in [YAMLKeyword.limit_opencl_kernel_time,
                    YAMLKeyword.nnlib_graph_mode,
                    YAMLKeyword.obfuscate,
                    YAMLKeyword.winograd,
                    YAMLKeyword.quantize,
                    YAMLKeyword.change_concat_ranges,
                    YAMLKeyword.model_type]:
            value = model_config.get(key, "")
            if value == "":
                model_config[key] = 0

        CONDITIONS(model_config[YAMLKeyword.winograd] in WinogradParameters,
                   ModuleName.YAML_CONFIG,
                   "'winograd' parameters must be in " + str(WinogradParameters) + ". 0 for disable winograd convolution")

        weight_file_path = model_config.get(YAMLKeyword.weight_file_path, "")
        model_config[YAMLKeyword.weight_file_path] = weight_file_path

    return configs


def clear_build_dirs(library_name):
    # make build dir
    if not os.path.exists(BUILD_OUTPUT_DIR):
        os.makedirs(BUILD_OUTPUT_DIR)
    # clear temp build dir
    tmp_build_dir = os.path.join(BUILD_OUTPUT_DIR, library_name,
                                 BUILD_TMP_DIR_NAME)
    if os.path.exists(tmp_build_dir):
        sh.rm('-rf', tmp_build_dir)
    os.makedirs(tmp_build_dir)
    # clear lib dir
    lib_output_dir = os.path.join(
        BUILD_OUTPUT_DIR, library_name, OUTPUT_LIBRARY_DIR_NAME)
    if os.path.exists(lib_output_dir):
        sh.rm('-rf', lib_output_dir)


################################
# convert
################################
def print_configuration(configs):
    title = "Common Configuration"
    header = ["key", "value"]
    data = list()
    data.append([YAMLKeyword.library_name,
                 configs[YAMLKeyword.library_name]])
    data.append([YAMLKeyword.target_abis,
                 configs[YAMLKeyword.target_abis]])
    data.append([YAMLKeyword.target_socs,
                 configs[YAMLKeyword.target_socs]])
    DLogger.summary(StringFormatter.table(header, data, title))


def download_file(url, dst, num_retries=3):
    from six.moves import urllib

    try:
        urllib.request.urlretrieve(url, dst)
        DLogger.info('\nDownloaded successfully.')
    except (urllib.error.ContentTooShortError, urllib.error.HTTPError,
            urllib.error.URLError) as e:
        DLogger.warning('Download error:' + str(e))
        if num_retries > 0:
            return download_file(url, dst, num_retries - 1)
        else:
            return False
    return True


def get_model_files(model_file_path,
                    model_output_dir,
                    weight_file_path="",
                    quantize_range_file_path=""):
    model_file = model_file_path
    weight_file = weight_file_path
    quantize_range_file = quantize_range_file_path

    if model_file_path.startswith("http://") or \
            model_file_path.startswith("https://"):
        model_file = model_output_dir + "/" + md5sum(model_file_path) + ".pb"
        if not os.path.exists(model_file):
            DLogger.info("Downloading model, please wait ...")
            if not download_file(model_file_path, model_file):
                DLogger.error(ModuleName.MODEL_CONVERTER,
                              "Model download failed.")

    if weight_file_path.startswith("http://") or \
            weight_file_path.startswith("https://"):
        weight_file = \
            model_output_dir + "/" + md5sum(weight_file_path) + ".caffemodel"
        if not os.path.exists(weight_file):
            DLogger.info("Downloading model weight, please wait ...")
            if not download_file(weight_file_path, weight_file):
                DLogger.error(ModuleName.MODEL_CONVERTER,
                              "Model download failed.")

    if quantize_range_file_path.startswith("http://") or \
            quantize_range_file_path.startswith("https://"):
        quantize_range_file = \
            model_output_dir + "/" + md5sum(quantize_range_file_path) \
            + ".range"
        if not download_file(quantize_range_file_path, quantize_range_file):
            DLogger.error(ModuleName.MODEL_CONVERTER,
                          "Model range file download failed.")
    return model_file, weight_file, quantize_range_file


def convert_model(configs, nuitka=False):
    # Remove previous output dirs
    library_name = configs[YAMLKeyword.library_name]
    if not os.path.exists(BUILD_OUTPUT_DIR):
        os.makedirs(BUILD_OUTPUT_DIR)
    elif os.path.exists(os.path.join(BUILD_OUTPUT_DIR, library_name)):
        sh.rm("-rf", os.path.join(BUILD_OUTPUT_DIR, library_name))
    os.makedirs(os.path.join(BUILD_OUTPUT_DIR, library_name))
    if not os.path.exists(BUILD_DOWNLOADS_DIR):
        os.makedirs(BUILD_DOWNLOADS_DIR)

    model_output_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)
    model_header_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_HEADER_DIR_PATH)
    # clear output dir
    if os.path.exists(model_output_dir):
        sh.rm("-rf", model_output_dir)
    os.makedirs(model_output_dir)
    if os.path.exists(model_header_dir):
        sh.rm("-rf", model_header_dir)

    if os.path.exists(MODEL_CODEGEN_DIR):
        sh.rm("-rf", MODEL_CODEGEN_DIR)
    if os.path.exists(ENGINE_CODEGEN_DIR):
        sh.rm("-rf", ENGINE_CODEGEN_DIR)

    for model_name in configs[YAMLKeyword.models]:
        DLogger.header(
            StringFormatter.block("Convert %s model" % model_name))
        model_config = configs[YAMLKeyword.models][model_name]
        runtime = model_config[YAMLKeyword.runtime]

        model_file_path, weight_file_path, quantize_range_file_path = \
            get_model_files(
                model_config[YAMLKeyword.model_file_path],
                BUILD_DOWNLOADS_DIR,
                model_config[YAMLKeyword.weight_file_path],
                model_config.get(YAMLKeyword.quantize_range_file, ""))

        data_type = model_config[YAMLKeyword.data_type]
        model_type = model_config[YAMLKeyword.model_type]
        # TODO@vgod: support multiple subgraphs
        subgraphs = model_config[YAMLKeyword.subgraphs]

        model_codegen_dir = "%s/%s" % (MODEL_CODEGEN_DIR, model_name)
        command_helper.gen_model_code(
            model_codegen_dir,
            model_config[YAMLKeyword.platform],
            model_file_path,
            weight_file_path,
            ",".join(subgraphs[0][YAMLKeyword.input_tensors]),
            ",".join(subgraphs[0][YAMLKeyword.input_data_types]),
            ",".join(subgraphs[0][YAMLKeyword.input_data_formats]),
            ",".join(subgraphs[0][YAMLKeyword.output_tensors]),
            ",".join(subgraphs[0][YAMLKeyword.output_data_types]),
            ",".join(subgraphs[0][YAMLKeyword.output_data_formats]),
            ",".join(subgraphs[0][YAMLKeyword.check_tensors]),
            runtime,
            model_name,
            ":".join(subgraphs[0][YAMLKeyword.input_shapes]),
            ":".join(subgraphs[0][YAMLKeyword.input_ranges]),
            ":".join(subgraphs[0][YAMLKeyword.output_shapes]),
            ":".join(subgraphs[0][YAMLKeyword.check_shapes]),
            model_config[YAMLKeyword.nnlib_graph_mode],
            model_config[YAMLKeyword.winograd],
            model_config[YAMLKeyword.quantize],
            quantize_range_file_path,
            get_pattern_weight(configs),
            get_pattern_config_path(configs),
            get_pattern_style_count(configs),
            get_conv_unroll(configs),
            model_config[YAMLKeyword.change_concat_ranges],
            model_config[YAMLKeyword.obfuscate],
            data_type,
            model_type,
            model_config[YAMLKeyword.cl_mem_type],
            ",".join(model_config.get(YAMLKeyword.graph_optimize_options, [])),
            is_load_from_im(configs),
            get_pruning_type(configs),
            get_executing_devices(configs),
            nuitka)

        sh.mv("-f",
              '%s/%s.pb' % (model_codegen_dir, model_name),
              model_output_dir)

        sh.mv("-f",
              '%s/%s.data' % (model_codegen_dir, model_name),
              model_output_dir)

        DLogger.summary(
            StringFormatter.block("Model %s converted" % model_name))


def print_library_summary(configs):
    library_name = configs[YAMLKeyword.library_name]
    title = "Library"
    header = ["key", "value"]
    data = list()
    data.append(["DeepVan Model Path",
                 "%s/%s/%s"
                 % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)])

    DLogger.summary(StringFormatter.table(header, data, title))


def convert_func(flags):
    configs = format_model_config(flags)
    print_configuration(configs)
    convert_model(configs, flags.nuitka)
    print_library_summary(configs)


################################
# run
################################
def build_deepvan_run(configs, target_abi, toolchain, enable_openmp,
                      address_sanitizer, deepvan_lib_type, debug_enable,
                      enable_xgen_fallback=False, static_openmp=False):
    library_name = configs[YAMLKeyword.library_name]

    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
    if os.path.exists(build_tmp_binary_dir):
        sh.rm("-rf", build_tmp_binary_dir)
    os.makedirs(build_tmp_binary_dir)

    deepvan_run_target = DEEPVAN_RUN_STATIC_TARGET
    if deepvan_lib_type == DEEPVANLibType.dynamic:
        deepvan_run_target = DEEPVAN_RUN_DYNAMIC_TARGET
    build_arg = ""

    command_helper.bazel_build(
        deepvan_run_target,
        abi=target_abi,
        toolchain=toolchain,
        enable_hexagon=get_hexagon_mode(configs),
        enable_hta=get_hta_mode(configs),
        enable_openmp=enable_openmp,
        enable_opencl=get_opencl_mode(configs),
        enable_quantize=get_quantize_mode(configs),
        enable_fallback=enable_xgen_fallback,
        address_sanitizer=address_sanitizer,
        symbol_hidden=get_symbol_hidden_mode(debug_enable, deepvan_lib_type),
        debug_enable=debug_enable,
        static_openmp=static_openmp,
        extra_args=build_arg
    )
    command_helper.update_deepvan_run_binary(build_tmp_binary_dir,
                                             deepvan_lib_type == DEEPVANLibType.dynamic)


def print_package_summary(package_path):
    title = "Library"
    header = ["key", "value"]
    data = list()
    data.append(["DeepVan Model package Path",
                 package_path])

    DLogger.summary(StringFormatter.table(header, data, title))


def gen_kernel_code(configs):
    if get_pattern_weight(configs) == 0:
        return
    pattern_config_path = get_pattern_config_path(configs)
    runtime_list = []
    for model_name in configs[YAMLKeyword.models]:
        model_runtime = configs[YAMLKeyword.models][model_name].get(
            YAMLKeyword.runtime, "")
        runtime_list.append(model_runtime.lower())
    if 'cpu' in runtime_list:
        sh.python3(
            "deepvan/tuning/python/source_code_generator_v2.py",
            "--config=%s" % pattern_config_path,
            "--save=%s" % is_save_pattern_kernel(configs),
        )
    else:
        sh.python3(
            "deepvan/tuning/python/source_code_generator_v2.py",
            "--config=%s" % pattern_config_path,
            "--save=%s" % is_save_pattern_kernel(configs),
        )


def generate_cmd_file(cmd, extra_args, prefix, model_tag, suffix):
    if extra_args:
        cmd.extend(extra_args)
    cmd = ' '.join(cmd) + ' $*'
    cmd_file_name = f'{prefix}-{model_tag}-{suffix}'
    cmd_file = f'{PHONE_DATA_DIR}/{cmd_file_name}'
    tmp_cmd_file = f'/tmp/{cmd_file_name}'
    with open(tmp_cmd_file, 'w') as file:
        file.write(cmd)
    return tmp_cmd_file


def generate(flags, configs, target_abi, pruning_type=PruningType.DENSE):

    # pointpillar_gpu 0 builds/pointpillar_gpu/_tmp/arm64-v8a
    library_name = configs[YAMLKeyword.library_name]
    deepvan_lib_type = flags.deepvan_lib_type
    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)

    # get target name for run
    if deepvan_lib_type == DEEPVANLibType.static:
        target_name = DEEPVAN_RUN_STATIC_NAME
    else:
        target_name = DEEPVAN_RUN_DYNAMIC_NAME
    link_dynamic = deepvan_lib_type == DEEPVANLibType.dynamic
    model_output_dirs = []

    for model_name in configs[YAMLKeyword.models]:
        check_model_converted(library_name, model_name, target_abi)

        model_config = configs[YAMLKeyword.models][model_name]
        model_runtime = model_config[YAMLKeyword.runtime]
        subgraphs = model_config[YAMLKeyword.subgraphs]

        # builds/pointpillar_gpu/_tmp/pointpillar_gpu/d8b9da25f253267bc46308b1d773acec/M2011K2C_lahaina/arm64-v8a
        # builds/pointpillar_gpu/model

        model_output_dir = os.path.join(
            "builds", model_name, "_tmp", target_abi)
        deepvan_model_dir = os.path.join("builds", model_name, "model")

        # clear temp model output dir
        if os.path.exists(model_output_dir):
            sh.rm('-rf', model_output_dir)
        os.makedirs(model_output_dir)

        command_helper.gen_random_input(
            model_output_dir,
            subgraphs[0][YAMLKeyword.input_tensors],
            subgraphs[0][YAMLKeyword.input_shapes],
            subgraphs[0][YAMLKeyword.validation_inputs_data],
            input_ranges=subgraphs[0][YAMLKeyword.input_ranges],
            input_data_types=subgraphs[0][YAMLKeyword.input_data_types]
        )
        runtime_list = []
        if target_abi == ABIType.host:
            runtime_list.append(RuntimeType.cpu)
        elif model_runtime == RuntimeType.cpu_gpu:
            runtime_list.extend([RuntimeType.cpu, RuntimeType.gpu])
        else:
            runtime_list.append(model_runtime)

        for runtime in runtime_list:
            device_type = parse_device_type(runtime)

            # run for specified soc
            if not subgraphs[0][YAMLKeyword.check_tensors]:
                output_nodes = subgraphs[0][YAMLKeyword.output_tensors]
                output_shapes = subgraphs[0][YAMLKeyword.output_shapes]
            else:
                output_nodes = subgraphs[0][YAMLKeyword.check_tensors]
                output_shapes = subgraphs[0][YAMLKeyword.check_shapes]
            output_configs = []
            log_file = ""
            if flags.layers != "-1":
                output_configs = DeviceWrapper.get_layers(deepvan_model_dir,
                                                          model_name,
                                                          flags.layers)
                log_dir = deepvan_model_dir + "/" + runtime
                if os.path.exists(log_dir):
                    sh.rm('-rf', log_dir)
                os.makedirs(log_dir)
                log_file = log_dir + "/log.csv"
            model_path = "%s/%s.pb" % (deepvan_model_dir, model_name)
            output_config = {YAMLKeyword.model_file_path: model_path,
                             YAMLKeyword.output_tensors: output_nodes,
                             YAMLKeyword.output_shapes: output_shapes}
            output_configs.append(output_config)

            model_opencl_output_bin_path = " "
            model_opencl_parameter_path = " "
            tuning_run(
                abi=target_abi,
                target_dir=build_tmp_binary_dir,
                target_name=target_name,
                vlog_level=flags.vlog_level,
                model_output_dir=model_output_dir,
                input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
                output_nodes=output_config[
                    YAMLKeyword.output_tensors],
                input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
                output_shapes=output_config[YAMLKeyword.output_shapes],
                input_data_formats=subgraphs[0][
                    YAMLKeyword.input_data_formats],
                output_data_formats=subgraphs[0][
                    YAMLKeyword.output_data_formats],
                deepvan_model_dir=deepvan_model_dir,
                model_tag=model_name,
                device_type=device_type,
                running_round=flags.round,
                warmup_running_round=flags.warmup_round,
                attempt_round=flags.attempt_round,
                restart_round=flags.restart_round,
                limit_opencl_kernel_time=model_config[
                    YAMLKeyword.limit_opencl_kernel_time],
                tuning=False,
                out_of_range_check=flags.gpu_out_of_range_check,
                omp_num_threads=flags.omp_num_threads,
                cpu_affinity_policy=flags.cpu_affinity_policy,
                gpu_perf_hint=flags.gpu_perf_hint,
                gpu_priority_hint=flags.gpu_priority_hint,
                address_sanitizer=flags.address_sanitizer,
                opencl_binary_file=model_opencl_output_bin_path,
                opencl_parameter_file=model_opencl_parameter_path,
                executor_dynamic_library_path=LIBDEEPVAN_DYNAMIC_PATH,
                link_dynamic=link_dynamic,
                quantize_stat=flags.quantize_stat,
                input_dir=flags.input_dir,
                output_dir=flags.output_dir,
                layers_validate_file=output_config[
                    YAMLKeyword.model_file_path],
                pruning_type=pruning_type,
                perf=flags.perf,
                generate_xgen_test_artifacts=flags.generate_xgen_test_artifacts,
                quantized=flags.generate_quantize_cmd,
                ignore_latency_outliers=flags.ignore_latency_outliers,
                benchmark=flags.benchmark,
            )
        print("Generated model and library in {}".format(
            "./generated/deepvan_run/"))


def run_deepvan(flags):
    configs = format_model_config(flags)
    gen_kernel_code(configs)

    clear_build_dirs(configs[YAMLKeyword.library_name])

    if flags.generate or flags.generate_xgen_test_artifacts:
        target_abi = "arm64-v8a"
        toolchain = "android"

        # For XGen, deepvan_run is built offline.
        if not flags.generate_xgen_test_artifacts:
            build_deepvan_run(configs,
                              target_abi,
                              toolchain,
                              not flags.disable_openmp,
                              flags.address_sanitizer,
                              flags.deepvan_lib_type,
                              flags.debug_enable)

        generate(flags, configs, target_abi, get_pruning_type(configs))
        return None

    target_socs = configs[YAMLKeyword.target_socs]
    device_list = DeviceManager.list_devices(flags.device_yml)
    if target_socs and TargetSOCTag.all not in target_socs:
        device_list = [
            dev for dev in device_list if dev[YAMLKeyword.target_socs].lower() in target_socs]
    for target_abi in configs[YAMLKeyword.target_abis]:
        if flags.target_socs == TargetSOCTag.random:
            target_devices = command_helper.choose_a_random_device(
                device_list, target_abi)
        else:
            target_devices = device_list
        target_devices = select_devices(flags, target_devices)
        # build target
        for dev in target_devices:
            if target_abi in dev[YAMLKeyword.target_abis]:
                # get toolchain
                toolchain = infer_toolchain(target_abi)
                device = DeviceWrapper(dev)
                build_deepvan_run(configs,
                                  target_abi,
                                  toolchain,
                                  not flags.disable_openmp,
                                  flags.address_sanitizer,
                                  flags.deepvan_lib_type,
                                  flags.debug_enable,
                                  flags.enable_xgen_fallback,
                                  flags.static_openmp)
                # run
                start_time = time.time()
                with device.lock():
                    device.run_specify_abi(
                        flags, configs, target_abi, get_pruning_type(configs))
                elapse_minutes = (time.time() - start_time) / 60
                print("Elapse time: %f minutes." % elapse_minutes)
            elif dev[YAMLKeyword.device_name] != SystemType.host:
                six.print_('The device with soc %s do not support abi %s' % (dev[YAMLKeyword.target_socs], target_abi),
                           file=sys.stderr)

    # package the output files
    # package_path = command_helper.packaging_lib(BUILD_OUTPUT_DIR,
    #                                          configs[YAMLKeyword.library_name])
    # print_package_summary(package_path)


################################
#  benchmark model
################################
def build_benchmark_model(configs,
                          target_abi,
                          toolchain,
                          enable_openmp,
                          deepvan_lib_type,
                          debug_enable,
                          static_openmp=False):
    library_name = configs[YAMLKeyword.library_name]

    link_dynamic = deepvan_lib_type == DEEPVANLibType.dynamic
    if link_dynamic:
        benchmark_target = BM_MODEL_DYNAMIC_TARGET
    else:
        benchmark_target = BM_MODEL_STATIC_TARGET

    build_arg = ""
    command_helper.bazel_build(benchmark_target,
                               abi=target_abi,
                               toolchain=toolchain,
                               enable_openmp=enable_openmp,
                               enable_opencl=get_opencl_mode(configs),
                               enable_quantize=get_quantize_mode(configs),
                               enable_hexagon=get_hexagon_mode(configs),
                               enable_hta=get_hta_mode(configs),
                               symbol_hidden=get_symbol_hidden_mode(debug_enable, deepvan_lib_type),  # noqa
                               debug_enable=debug_enable,
                               static_openmp=static_openmp,
                               extra_args=build_arg)
    # clear tmp binary dir
    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
    if os.path.exists(build_tmp_binary_dir):
        sh.rm("-rf", build_tmp_binary_dir)
    os.makedirs(build_tmp_binary_dir)

    target_bin = "/".join(command_helper.bazel_target_to_bin(benchmark_target))
    sh.cp("-f", target_bin, build_tmp_binary_dir)


def select_devices(flags, available_devices):
    target_devices = list()
    for target in flags.target_devices.split(":"):
        for available in available_devices:
            if target in str(available):
                target_devices.append(available)
    return available_devices if len(target_devices) == 0 else target_devices


def benchmark_deepvan(flags):
    configs = format_model_config(flags)
    gen_kernel_code(configs)

    clear_build_dirs(configs[YAMLKeyword.library_name])

    target_socs = configs[YAMLKeyword.target_socs]
    device_list = DeviceManager.list_devices(flags.device_yml)
    if target_socs and TargetSOCTag.all not in target_socs:
        device_list = [
            dev for dev in device_list if dev[YAMLKeyword.target_socs].lower() in target_socs]
    for target_abi in configs[YAMLKeyword.target_abis]:
        if flags.target_socs == TargetSOCTag.random:
            target_devices = command_helper.choose_a_random_device(
                device_list, target_abi)
        else:
            target_devices = device_list
        # build benchmark_model binary
        result = []
        target_devices = select_devices(flags, target_devices)
        for dev in target_devices:
            six.print_(CMDColors.PURPLE + "Evaluating on device: %s" %
                       (dev["device_name"]) + CMDColors.ENDC)
            if target_abi in dev[YAMLKeyword.target_abis]:
                toolchain = infer_toolchain(target_abi)
                build_benchmark_model(configs,
                                      target_abi,
                                      toolchain,
                                      not flags.disable_openmp,
                                      flags.deepvan_lib_type,
                                      flags.debug_enable,
                                      flags.static_openmp)
                device = DeviceWrapper(dev)
                start_time = time.time()
                with device.lock():
                    result = device.bm_specific_target(flags,
                                                       configs,
                                                       target_abi,
                                                       is_tuning_mode(configs),
                                                       get_pruning_type(configs))
                elapse_minutes = (time.time() - start_time) / 60
                print("Elapse time: %f minutes." % elapse_minutes)
            else:
                six.print_('There is no abi %s with soc %s' % (target_abi, dev[YAMLKeyword.target_socs]),
                           file=sys.stderr)
        return result


################################
# parsing arguments
################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_to_deepvan_lib_type(v):
    if v.lower() == 'dynamic':
        return DEEPVANLibType.dynamic
    elif v.lower() == 'static':
        return DEEPVANLibType.static
    else:
        raise argparse.ArgumentTypeError('[dynamic| static] expected.')


def parse_args():
    """Parses command line arguments."""
    all_type_parent_parser = argparse.ArgumentParser(add_help=False)
    # only for pointpillar
    all_type_parent_parser.add_argument(
        '--model_path',
        type=str,
        default="",
        required=False,
        help="the path of model file.")
    all_type_parent_parser.add_argument(
        '--config',
        type=str,
        default="",
        required=True,
        help="the path of model yaml configuration file.")
    all_type_parent_parser.add_argument(
        "--target_abis",
        type=str,
        default="",
        help="Target ABIs, comma seperated list.")
    all_type_parent_parser.add_argument(
        "--target_socs",
        type=str,
        default="",
        help="Target SOCs, comma seperated list.")
    all_type_parent_parser.add_argument(
        "--target_devices",
        type=str,
        default="",
        help="Target Devices, comma seperated list.")
    all_type_parent_parser.add_argument(
        "--debug_enable",
        action="store_true",
        help="Reserve debug symbols.")
    convert_run_parent_parser = argparse.ArgumentParser(add_help=False)
    convert_run_parent_parser.add_argument(
        '--address_sanitizer',
        action="store_true",
        help="Whether to use address sanitizer to check memory error")
    convert_run_parent_parser.add_argument(
        '--nuitka',
        action="store_true",
        help="Whether to use nuitka binary to run convert")
    run_bm_parent_parser = argparse.ArgumentParser(add_help=False)
    run_bm_parent_parser.add_argument(
        "--deepvan_lib_type",
        type=str_to_deepvan_lib_type,
        default=DefaultValues.deepvan_lib_type,
        help="[static | dynamic], Which type Deepvan library to use.")
    run_bm_parent_parser.add_argument(
        "--disable_openmp",
        action="store_true",
        help="Disable openmp for multiple thread.")
    run_bm_parent_parser.add_argument(
        "--omp_num_threads",
        type=int,
        default=DefaultValues.omp_num_threads,
        help="num of openmp threads")
    run_bm_parent_parser.add_argument(
        "--cpu_affinity_policy",
        type=int,
        default=DefaultValues.cpu_affinity_policy,
        help="0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY")
    run_bm_parent_parser.add_argument(
        "--gpu_perf_hint",
        type=int,
        default=DefaultValues.gpu_perf_hint,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")
    run_bm_parent_parser.add_argument(
        "--gpu_priority_hint",
        type=int,
        default=DefaultValues.gpu_priority_hint,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")
    run_bm_parent_parser.add_argument(
        "--device_yml",
        type=str,
        default='',
        help='embedded linux device config yml file'
    )
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    convert = subparsers.add_parser(
        'convert',
        parents=[all_type_parent_parser, convert_run_parent_parser],
        help='convert to deepvan model (file or code)')
    convert.set_defaults(func=convert_func)
    run = subparsers.add_parser(
        'run',
        parents=[all_type_parent_parser, run_bm_parent_parser,
                 convert_run_parent_parser],
        help='run model in command line')
    run.set_defaults(func=run_deepvan)
    run.add_argument(
        "--disable_tuning",
        action="store_true",
        help="Disable tuning for specific thread.")
    run.add_argument(
        "--round",
        type=int,
        default=1,
        help="The model running round.")
    run.add_argument(
        "--warmup_round",
        type=int,
        default=1,
        help="The model warmup running round.")
    run.add_argument(
        "--attempt_round",
        type=int,
        default=0,
        help="The model running round after failure. 0 means always retry.")
    run.add_argument(
        "--validate",
        action="store_true",
        help="whether to verify the results are consistent with "
             "the frameworks.")
    run.add_argument(
        "--perf",
        action="store_true",
        help="whether to use simpleperf to profile the running.")
    run.add_argument(
        "--layers",
        type=str,
        default="-1",
        help="'start_layer:end_layer' or 'layer', similar to python slice."
             " Use with --validate flag.")
    run.add_argument(
        "--vlog_level",
        type=int,
        default=0,
        help="[1~5]. Verbose log level for debug.")
    run.add_argument(
        "--gpu_out_of_range_check",
        action="store_true",
        help="Enable out of memory check for gpu.")
    run.add_argument(
        "--restart_round",
        type=int,
        default=1,
        help="restart round between run.")
    run.add_argument(
        "--report",
        action="store_true",
        help="print run statistics report.")
    run.add_argument(
        "--report_dir",
        type=str,
        default="",
        help="print run statistics report.")
    run.add_argument(
        "--quantize_stat",
        action="store_true",
        help="whether to stat quantization range.")
    run.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="quantize stat input dir.")
    run.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="quantize stat output dir.")
    run.add_argument(
        "--generate",
        action="store_true",
        help="Generate code without connected devices.")
    run.add_argument(
        "--generate_xgen_test_artifacts",
        action="store_true",
        help="generate test artifacts for XGen.")
    run.add_argument(
        "--enable_xgen_fallback",
        action="store_true",
        help="integrate Tensorflow Lite into deepvan_run_static as fallback for XGen.")
    run.add_argument(
        "--generate_quantize_cmd",
        action="store_true",
        help="generate cmd file with --quantized option for deepvan_run_static.")
    run.add_argument(
        "--static_openmp",
        action="store_true",
        help="link openmp statically for NDK without libomp.so.")
    run.add_argument(
        "--ignore_latency_outliers",
        action="store_true",
        help="ignore outliers when computing average latency.")
    run.add_argument(
        "--benchmark_offline",
        action="store_true",
        help="benchmark model for detail information.")
    
    return parser.parse_known_args()


if __name__ == "__main__":
    flags, unparsed = parse_args()
    flags.func(flags)
