import enum
import hashlib
import inspect
import os

import six

################################
# log
################################


class CMDColors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_frame_info(level=2):
    caller_frame = inspect.stack()[level]
    info = inspect.getframeinfo(caller_frame[0])
    return info.filename + ':' + str(info.lineno) + ': '


class DLogger:
    @staticmethod
    def header(message):
        six.print_(CMDColors.PURPLE + message + CMDColors.ENDC)

    @staticmethod
    def summary(message):
        six.print_(CMDColors.GREEN + message + CMDColors.ENDC)

    @staticmethod
    def info(message):
        six.print_(get_frame_info() + message)

    @staticmethod
    def warning(message):
        six.print_(CMDColors.YELLOW + 'WARNING:' + get_frame_info() + message +
                   CMDColors.ENDC)

    @staticmethod
    def error(module, message, location_info=""):
        if not location_info:
            location_info = get_frame_info()
        six.print_(CMDColors.RED + 'ERROR: [' + module + '] ' + location_info +
                   message + CMDColors.ENDC)
        exit(1)


def CONDITIONS(condition, module, message):
    if not condition:
        DLogger.error(module, message, get_frame_info())


################################
# String Formatter
################################
class StringFormatter:
    @staticmethod
    def table(header, data, title, align="R"):
        data_size = len(data)
        column_size = len(header)
        column_length = [len(str(ele)) + 1 for ele in header]
        for row_idx in range(data_size):
            data_tuple = data[row_idx]
            ele_size = len(data_tuple)
            assert (ele_size == column_size)
            for i in range(ele_size):
                column_length[i] = max(column_length[i],
                                       len(str(data_tuple[i])) + 1)

        table_column_length = sum(column_length) + column_size + 1
        dash_line = '-' * table_column_length + '\n'
        header_line = '=' * table_column_length + '\n'
        output = ""
        output += dash_line
        output += str(title).center(table_column_length) + '\n'
        output += dash_line
        output += '|' + '|'.join([str(header[i]).center(column_length[i])
                                 for i in range(column_size)]) + '|\n'
        output += header_line

        for data_tuple in data:
            ele_size = len(data_tuple)
            row_list = []
            for i in range(ele_size):
                if align == "R":
                    row_list.append(str(data_tuple[i]).rjust(column_length[i]))
                elif align == "L":
                    row_list.append(str(data_tuple[i]).ljust(column_length[i]))
                elif align == "C":
                    row_list.append(str(data_tuple[i])
                                    .center(column_length[i]))
            output += '|' + '|'.join(row_list) + "|\n" + dash_line
        return output

    @staticmethod
    def block(message):
        line_length = 10 + len(str(message)) + 10
        star_line = '*' * line_length + '\n'
        return star_line + str(message).center(line_length) + '\n' + star_line


################################
# definitions
################################
class DeviceType(object):
    CPU = 'CPU'
    GPU = 'GPU'
    HEXAGON = 'HEXAGON'
    HTA = 'HTA'


class DataFormat(object):
    NONE = "NONE"
    NHWC = "NHWC"
    NCHW = "NCHW"
    OIHW = "OIHW"


################################
# Argument types
################################
class CaffeEnvType(enum.Enum):
    DOCKER = 0,
    LOCAL = 1,


################################
# common functions
################################
def formatted_file_name(input_file_name, input_name):
    res = input_file_name + '_'
    for c in input_name:
        res += c if c.isalnum() else '_'
    return res


def md5sum(s):
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()


def get_build_binary_dir(library_name, target_abi):
    return "%s/%s/%s/%s" % (
        BUILD_OUTPUT_DIR, library_name, BUILD_TMP_DIR_NAME, target_abi)


def get_model_lib_output_path(library_name, abi):
    lib_output_path = os.path.join(BUILD_OUTPUT_DIR, library_name,
                                   MODEL_OUTPUT_DIR_NAME, abi,
                                   "%s.a" % library_name)
    return lib_output_path


def check_model_converted(library_name, model_name, abi):
    model_output_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)
    CONDITIONS(os.path.exists("%s/%s.pb" % (model_output_dir, model_name)),
               ModuleName.RUN,
               "You should convert model first.")
    CONDITIONS(os.path.exists("%s/%s.data" %
                              (model_output_dir, model_name)),
               ModuleName.RUN,
               "You should convert model first.")


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


def sha256_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_model_files(model_file_path,
                    model_output_dir,
                    weight_file_path=""):
    model_file = model_file_path
    weight_file = weight_file_path

    if model_file_path.startswith("http://") or \
            model_file_path.startswith("https://"):
        model_file = model_output_dir + "/" + md5sum(model_file_path) + ".pb"
        if not os.path.exists(model_file):
            DLogger.info("Downloading model, please wait ...")
            six.moves.urllib.request.urlretrieve(model_file_path, model_file)
            DLogger.info("Model downloaded successfully.")

    if weight_file_path.startswith("http://") or \
            weight_file_path.startswith("https://"):
        weight_file = \
            model_output_dir + "/" + md5sum(weight_file_path) + ".caffemodel"
        if not os.path.exists(weight_file):
            DLogger.info("Downloading model weight, please wait ...")
            six.moves.urllib.request.urlretrieve(weight_file_path, weight_file)
            DLogger.info("Model weight downloaded successfully.")

    if weight_file:
        pass

    return model_file, weight_file


def get_build_model_dirs(library_name,
                         model_name,
                         target_abi,
                         device,
                         model_file_path):
    device_name = device.device_name
    target_socs = device.target_socs
    model_path_digest = md5sum(model_file_path)
    model_output_base_dir = '{}/{}/{}/{}/{}'.format(
        BUILD_OUTPUT_DIR, library_name, BUILD_TMP_DIR_NAME,
        model_name, model_path_digest)

    if target_abi == ABIType.host:
        model_output_dir = '%s/%s' % (model_output_base_dir, target_abi)
    elif not target_socs or not device.address:
        model_output_dir = '%s/%s/%s' % (model_output_base_dir,
                                         BUILD_TMP_GENERAL_OUTPUT_DIR_NAME,
                                         target_abi)
    else:
        model_output_dir = '{}/{}_{}/{}'.format(
            model_output_base_dir,
            device_name,
            target_socs,
            target_abi
        )

    model_dir = '{}/{}/{}'.format(
        BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME
    )

    return model_output_base_dir, model_output_dir, model_dir


def abi_to_internal(abi):
    if abi in [ABIType.armeabi_v7a, ABIType.arm64_v8a]:
        return abi
    if abi == ABIType.arm64:
        return ABIType.aarch64
    if abi == ABIType.armhf:
        return ABIType.armeabi_v7a


def infer_toolchain(abi):
    if abi in [ABIType.armeabi_v7a, ABIType.arm64_v8a]:
        return ToolchainType.android
    if abi == ABIType.armhf:
        return ToolchainType.arm_linux_gnueabihf
    if abi == ABIType.arm64:
        return ToolchainType.aarch64_linux_gnu
    return ''


################################
# YAML key word
################################
class YAMLKeyword(object):
    library_name = 'library_name'
    target_abis = 'target_abis'
    target_socs = 'target_socs'
    models = 'models'
    platform = 'platform'
    device_name = 'device_name'
    system = 'system'
    address = 'address'
    username = 'username'
    password = 'password'
    model_file_path = 'model_file_path'
    validate_model_file_path = 'validate_model_file_path'
    weight_file_path = 'weight_file_path'
    subgraphs = 'subgraphs'
    input_tensors = 'input_tensors'
    input_shapes = 'input_shapes'
    input_ranges = 'input_ranges'
    output_tensors = 'output_tensors'
    output_shapes = 'output_shapes'
    check_tensors = 'check_tensors'
    check_shapes = 'check_shapes'
    runtime = 'runtime'
    load_from_im = 'load_from_im'
    executing_devices = 'executing_only'
    data_type = 'data_type'
    model_type = 'model_type'
    input_data_types = 'input_data_types'
    output_data_types = 'output_data_types'
    input_data_formats = 'input_data_formats'
    output_data_formats = 'output_data_formats'
    limit_opencl_kernel_time = 'limit_opencl_kernel_time'
    nnlib_graph_mode = 'nnlib_graph_mode'
    obfuscate = 'obfuscate'
    winograd = 'winograd'
    quantize = 'quantize'
    pattern_weight = 'pattern_weight'
    pattern_config_path = 'pattern_config_path'
    pattern_style_count = 'pattern_style_count'
    save_pattern_kernel = 'save_pattern_kernel'
    tuning_mode = 'tuning_mode'
    sparsed_weight = 'sparsed_weight'
    quantize_range_file = 'quantize_range_file'
    change_concat_ranges = 'change_concat_ranges'
    validation_inputs_data = 'validation_inputs_data'
    validation_threshold = 'validation_threshold'
    graph_optimize_options = 'graph_optimize_options'  # internal use for now
    cl_mem_type = 'cl_mem_type'
    backend = 'backend'
    validation_outputs_data = 'validation_outputs_data'
    pruning_type = 'pruning_type'
    conv_unroll = 'conv_unroll'


################################
# SystemType
################################
class SystemType:
    host = 'host'
    android = 'android'
    arm_linux = 'arm_linux'


################################
# common device str
################################

PHONE_DATA_DIR = '/data/local/tmp/deepvan_run'
DEVICE_DATA_DIR = '/tmp/data/deepvan_run'
DEVICE_INTERIOR_DIR = PHONE_DATA_DIR + "/interior"
BUILD_OUTPUT_DIR = 'builds'
BUILD_TMP_DIR_NAME = '_tmp'
BUILD_DOWNLOADS_DIR = BUILD_OUTPUT_DIR + '/downloads'
BUILD_TMP_GENERAL_OUTPUT_DIR_NAME = 'general'
MODEL_OUTPUT_DIR_NAME = 'model'
CL_COMPILED_BINARY_FILE_NAME = "deepvan_cl_compiled_program.bin"
BUILD_TMP_OPENCL_BIN_DIR = 'opencl_bin'
CL_TUNED_PARAMETER_FILE_NAME = "deepvan_run.config"
MODEL_HEADER_DIR_PATH = 'include/deepvan/export'
OUTPUT_LIBRARY_DIR_NAME = 'lib'
OUTPUT_OPENCL_BINARY_DIR_NAME = 'opencl'
OUTPUT_OPENCL_BINARY_FILE_NAME = 'compiled_opencl_kernel'
OUTPUT_OPENCL_PARAMETER_FILE_NAME = 'tuned_opencl_parameter'
CODEGEN_BASE_DIR = 'deepvan/codegen'
MODEL_CODEGEN_DIR = CODEGEN_BASE_DIR + '/models'
ENGINE_CODEGEN_DIR = CODEGEN_BASE_DIR + '/engine'
LIB_CODEGEN_DIR = CODEGEN_BASE_DIR + '/lib'
OPENCL_CODEGEN_DIR = CODEGEN_BASE_DIR + '/opencl'
QUANTIZE_STAT_TARGET = "//deepvan/tools/quantization:quantize_stat"
# generated model target
MODEL_LIB_TARGET = "//deepvan/codegen:generated_models"
MODEL_LIB_PATH = "bazel-genfiles/deepvan/codegen/libgenerated_models.a"
# deepvan lib target
LIBDEEPVAN_DYNAMIC_PATH = "bazel-bin/deepvan/executor/libexecutor_shared.so"
LIBDEEPVAN_SO_TARGET = "//deepvan/executor:libexecutor_shared.so"
LIBDEEPVAN_STATIC_TARGET = "//deepvan/executor:executor_static"
LIBDEEPVAN_STATIC_PATH = "bazel-genfiles/deepvan/executor/executor.a"
# run dynamic/static target
DEEPVAN_RUN_STATIC_NAME = "deepvan_run_static"
DEEPVAN_RUN_DYNAMIC_NAME = "deepvan_run_dynamic"
DEEPVAN_RUN_STATIC_TARGET = "//deepvan/run:" + DEEPVAN_RUN_STATIC_NAME
DEEPVAN_RUN_DYNAMIC_TARGET = "//deepvan/run:" + DEEPVAN_RUN_DYNAMIC_NAME
# benchmark dynamic/static target
BM_MODEL_STATIC_NAME = "benchmark_model_static"
BM_MODEL_DYNAMIC_NAME = "benchmark_model_dynamic"
BM_MODEL_STATIC_TARGET = "//deepvan/run:" + BM_MODEL_STATIC_NAME
BM_MODEL_DYNAMIC_TARGET = "//deepvan/run:" + BM_MODEL_DYNAMIC_NAME

################################
# ABI Type
################################


class ABIType(object):
    armeabi_v7a = 'armeabi-v7a'
    arm64_v8a = 'arm64-v8a'
    arm64 = 'arm64'
    aarch64 = 'aarch64'
    armhf = 'armhf'
    host = 'host'


################################
# Module name
################################
class ModuleName(object):
    YAML_CONFIG = 'YAML CONFIG'
    MODEL_CONVERTER = 'Model Converter'
    RUN = 'RUN'
    BENCHMARK = 'Benchmark'


#################################
# deepvan lib type
#################################
class DEEPVANLibType(object):
    static = 0
    dynamic = 1


#################################
# Run time type
#################################
class RuntimeType(object):
    cpu = 'cpu'
    gpu = 'gpu'
    dsp = 'dsp'
    hta = 'hta'
    cpu_gpu = 'cpu+gpu'


#################################
# Tool chain Type
#################################
class ToolchainType:
    android = 'android'
    arm_linux_gnueabihf = 'arm_linux_gnueabihf'
    aarch64_linux_gnu = 'aarch64_linux_gnu'


#################################
# SOC tag
#################################
class TargetSOCTag:
    all = 'all'
    random = 'random'


def split_shape(shape):
    if shape.strip() == "":
        return []
    else:
        return shape.split(',')


class PruningType:
    DENSE = "DENSE"
