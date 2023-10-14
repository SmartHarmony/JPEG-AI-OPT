from enum import Enum
import time
import six
# from collections import abc
import sys
from deepvan.proto import deepvan_pb2


class DeviceType(Enum):
    CPU = 0
    GPU = 2
    HEXAGON = 3
    HTA = 4
    APU = 5


ConstantOperation = [
    "Shape",
    "Gather",
    "Unsqueeze",
    "Concat",
    "Reshape"
]


class PruningType(Enum):
    DENSE = "DENSE",


class ModelType(Enum):
    default = "default"


class DataFormat(Enum):
    DF_NONE = 0
    NHWC = 1
    NCHW = 2
    HWIO = 100
    OIHW = 101
    HWOI = 102
    OHWI = 103
    AUTO = 1000


# SAME_LOWER: if the amount of paddings to be added is odd,
# it will add the extra data to the right or bottom
class PaddingMode(Enum):
    VALID = 0
    SAME = 1
    FULL = 2
    SAME_LOWER = 3
    NA = 4


class PaddingType:
    CONSTANT = 0
    REFLECT = 1
    SYMMETRIC = 2


class PoolingType(Enum):
    AVG = 1
    MAX = 2


class RoundMode(Enum):
    FLOOR = 0
    CEIL = 1


class ActivationType(Enum):
    NOOP = 0
    RELU = 1
    RELUX = 2
    PRELU = 3
    TANH = 4
    SIGMOID = 5
    LEAKYRELU = 6
    ROUND = 7
    HARDSIGMOID = 8


class EltwiseType(Enum):
    SUM = 0
    SUB = 1
    PROD = 2
    DIV = 3
    MIN = 4
    MAX = 5
    NEG = 6
    ABS = 7
    SQR_DIFF = 8
    POW = 9
    EQUAL = 10
    FLOOR_DIV = 11
    CLIP = 12
    FLOOR = 13
    EXP = 14
    Erf = 15


class ReduceType(Enum):
    MEAN = 0
    MIN = 1
    MAX = 2
    PROD = 3
    SUM = 4


class PadType(Enum):
    CONSTANT = 0
    REFLECT = 1
    SYMMETRIC = 2


class FrameworkType(Enum):
    TENSORFLOW = 0
    CAFFE = 1
    ONNX = 2


DeepVanSupportedOps = [
    'Activation',
    'Conv2D',
    'Deconv2D',
    'Eltwise',
    'Reshape',
    'Transpose',
]

DeepvanOp = Enum('DeepvanOp', [(op, op)
                               for op in DeepVanSupportedOps], type=str)

# means different data format may have different computation
DeepvanFixedDataFormatOps = [
    DeepvanOp.Conv2D,
    DeepvanOp.Deconv2D,
]

# means no matter what data format, the same computation
DeepvanTransposableDataFormatOps = [
    DeepvanOp.Activation,
    DeepvanOp.Eltwise,
    DeepvanOp.Transpose,
]


class DeepvanConfigKey(object):
    # node related str
    deepvan_input_node_name = 'deepvan_input_node'
    deepvan_output_node_name = 'deepvan_output_node'
    deepvan_buffer_type = 'buffer_type'
    # arg related str
    deepvan_padding_str = 'padding'
    deepvan_padding_type_str = 'pad_type'
    deepvan_padding_values_str = 'padding_values'
    deepvan_strides_str = 'strides'
    deepvan_dilations_str = 'dilations'
    deepvan_pooling_type_str = 'pooling_type'
    deepvan_global_pooling_str = 'global_pooling'
    deepvan_kernel_str = 'kernels'
    deepvan_data_format_str = 'data_format'
    deepvan_has_data_format_str = 'has_data_format'
    deepvan_filter_format_str = 'filter_format'
    deepvan_element_type_str = 'type'
    deepvan_activation_type_str = 'activation'
    deepvan_activation_max_limit_str = 'max_limit'
    deepvan_activation_leakyrelu_coefficient_str = 'leakyrelu_coefficient'
    deepvan_activation_hardsigmoid_beta = 'beta'
    deepvan_resize_size_str = 'size'
    deepvan_batch_to_space_crops_str = 'crops'
    deepvan_paddings_str = 'paddings'
    deepvan_align_corners_str = 'align_corners'
    deepvan_space_batch_block_shape_str = 'block_shape'
    deepvan_space_depth_block_size_str = 'block_size'
    deepvan_constant_value_str = 'constant_value'
    deepvan_dim_str = 'dim'
    deepvan_dims_str = 'dims'
    deepvan_axis_str = 'axis'
    deepvan_end_axis_str = 'end_axis'
    deepvan_num_axes_str = 'num_axes'
    deepvan_num_split_str = 'num_split'
    deepvan_keepdims_str = 'keepdims'
    deepvan_shape_str = 'shape'
    deepvan_winograd_filter_transformed = 'is_filter_transformed'
    deepvan_device = 'device'
    deepvan_scalar_input_str = 'scalar_input'
    deepvan_wino_block_size = 'wino_block_size'
    deepvan_output_shape_str = 'output_shape'
    deepvan_begin_mask_str = 'begin_mask'
    deepvan_end_mask_str = 'end_mask'
    deepvan_ellipsis_mask_str = 'ellipsis_mask'
    deepvan_new_axis_mask_str = 'new_axis_mask'
    deepvan_shrink_axis_mask_str = 'shrink_axis_mask'
    deepvan_transpose_a_str = 'transpose_a'
    deepvan_transpose_b_str = 'transpose_b'
    deepvan_op_data_type_str = 'T'
    deepvan_offset_str = 'offset'
    deepvan_opencl_max_image_size = "opencl_max_image_size"
    deepvan_seperate_buffer_str = 'seperate_buffer'
    deepvan_scalar_input_index_str = 'scalar_input_index'
    deepvan_opencl_mem_type = "opencl_mem_type"
    deepvan_framework_type_str = "framework_type"
    deepvan_group_str = "group"
    deepvan_wino_arg_str = "wino_block_size"
    # deepvan_quantize_flag_arg_str = "quantize_flag"
    deepvan_pattern_flag_arg_str = "pattern_flag"
    deepvan_sparse_format_arg_str = "sparse_format"
    deepvan_pruning_type_arg_str = "pruning_type"
    deepvan_model_type_arg_str = "model_type"
    deepvan_conv_unroll_arg_str = "conv_unroll"
    deepvan_epsilon_str = 'epsilon'
    deepvan_reduce_type_str = 'reduce_type'
    deepvan_argmin_str = 'argmin'
    deepvan_round_mode_str = 'round_mode'
    deepvan_upsample_mode = 'mode'
    deepvan_min_size_str = 'min_size'
    deepvan_max_size_str = 'max_size'
    deepvan_aspect_ratio_str = 'aspect_ratio'
    deepvan_flip_str = 'flip'
    deepvan_clip_str = 'clip'
    deepvan_variance_str = 'variance'
    deepvan_step_h_str = 'step_h'
    deepvan_step_w_str = 'step_w'
    deepvan_find_range_every_time = 'find_range_every_time'
    deepvan_non_zero = 'non_zero'
    deepvan_pad_type_str = 'pad_type'
    deepvan_exclusive_str = 'exclusive'
    deepvan_reverse_str = 'reverse'
    deepvan_const_data_num_arg_str = 'const_data_num'
    deepvan_coeff_str = 'coeff'
    deepvan_executing_on_str = 'executing_on'
    is_last_op = 'is_last_op'
    is_first_op = 'is_first_op'
    deepvan_scale_value_str = "scale"
    deepvan_output_padding_str = "output_padding"


class TransformerRule(Enum):
    ADD_WINOGRAD_ARG = 8  # KEEP
    TRANSPOSE_FILTERS = 13
    TRANSPOSE_DATA_FORMAT = 15  # KEEP
    SORT_BY_EXECUTION = 19  # KEEP
    ADD_IN_OUT_TENSOR_INFO = 20  # KEEP
    ADD_OPENCL_INFORMATIONS = 31  # KEEP
    UPDATE_DATA_FORMAT = 39  # KEEP


class ConverterInterface(object):
    """Base class for converting external models to deepvan models."""

    def run(self):
        raise NotImplementedError('run')


class NodeInfo(object):
    """A class for describing node information"""

    def __init__(self):
        self.name = None
        self.data_type = deepvan_pb2.DT_FLOAT
        self.shape = []
        self.data_format = DataFormat.NHWC
        self.range = [-1.0, 1.0]

    def __str__(self):
        return '%s %s %s %s' % (self.name, self.shape,
                                self.data_type, self.data_format)


class NodeMapProperty:
    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = "_{}#{}".format(prefix, index)
        cls.__counter += 1

    def __set__(self, instance, data):
        # if not isinstance(data, abc.Mapping):
        # 	raise ValueError("Value should be map")
        if not hasattr(instance, self.storage_name):
            setattr(instance, self.storage_name, {})
        for item in data.values():
            getattr(instance, self.storage_name)[item.name] = item

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.storage_name)


class ConverterOption(object):
    """A class for specifying options passed to converter tool"""
    input_nodes = NodeMapProperty()
    output_nodes = NodeMapProperty()
    check_nodes = NodeMapProperty()

    def __init__(self):
        self.input_nodes = {}
        self.output_nodes = {}
        self.check_nodes = {}
        self.data_type = deepvan_pb2.DT_FLOAT
        self.model_type = ""
        self.device = DeviceType.CPU.value
        self.winograd = 0
        self.change_concat_ranges = False
        self.transformer_option = None
        self.cl_mem_type = ""
        self.sparse_weight = False
        self.pruning_type = ""
        self.executing_devices = {}

    def add_input_node(self, input_node):
        self.input_nodes[input_node.name] = input_node

    def add_output_node(self, output_node):
        self.output_nodes[output_node.name] = output_node

    def add_check_node(self, check_node):
        self.check_nodes[check_node.name] = check_node

    def disable_transpose_filters(self):
        if TransformerRule.TRANSPOSE_FILTERS in self.transformer_option:
            self.transformer_option.remove(TransformerRule.TRANSPOSE_FILTERS)

    def enable_transpose_filters(self):
        if TransformerRule.TRANSPOSE_FILTERS not in self.transformer_option:
            self.transformer_option.append(TransformerRule.TRANSPOSE_FILTERS)

    def build(self):
        if self.transformer_option:
            self.transformer_option = [TransformerRule[transformer]
                                       for transformer in self.transformer_option]  # noqa
        else:
            self.transformer_option = [
                TransformerRule.TRANSPOSE_FILTERS,
                TransformerRule.ADD_IN_OUT_TENSOR_INFO,
                # Add winograd argument
                TransformerRule.ADD_WINOGRAD_ARG,
                # Transform finalization
                TransformerRule.ADD_OPENCL_INFORMATIONS,
                TransformerRule.SORT_BY_EXECUTION,
                # update the data format of ops
                TransformerRule.UPDATE_DATA_FORMAT,
                TransformerRule.TRANSPOSE_DATA_FORMAT,
            ]


class ConverterUtil(object):
    @staticmethod
    def get_arg(op, arg_name):
        for arg in op.arg:
            if arg.name == arg_name:
                return arg
        return None

    @staticmethod
    def del_arg(op, arg_name):
        found_idx = -1
        for idx in range(len(op.arg)):
            if op.arg[idx].name == arg_name:
                found_idx = idx
                break
        if found_idx != -1:
            del op.arg[found_idx]

    @staticmethod
    def add_data_format_arg(op, data_format):
        data_format_arg = op.arg.add()
        data_format_arg.name = DeepvanConfigKey.deepvan_data_format_str
        data_format_arg.i = data_format.value

    @staticmethod
    def add_data_type_arg(op, data_type):
        data_type_arg = op.arg.add()
        data_type_arg.name = DeepvanConfigKey.deepvan_op_data_type_str
        data_type_arg.i = data_type

    @staticmethod
    def data_format(op):
        arg = ConverterUtil.get_arg(
            op, DeepvanConfigKey.deepvan_data_format_str)
        if arg is None:
            return None
        elif arg.i == DataFormat.NHWC.value:
            return DataFormat.NHWC
        elif arg.i == DataFormat.NCHW.value:
            return DataFormat.NCHW
        elif arg.i == DataFormat.AUTO.value:
            return DataFormat.AUTO
        else:
            return None

    @staticmethod
    def set_filter_format(net, filter_format):
        arg = net.arg.add()
        arg.name = DeepvanConfigKey.deepvan_filter_format_str
        arg.i = filter_format.value

    @staticmethod
    def filter_format(net):
        arg = ConverterUtil.get_arg(
            net, DeepvanConfigKey.deepvan_filter_format_str)
        if arg is None:
            return None
        elif arg.i == DataFormat.HWIO.value:
            return DataFormat.HWIO
        elif arg.i == DataFormat.HWOI.value:
            return DataFormat.HWOI
        elif arg.i == DataFormat.OIHW.value:
            return DataFormat.OIHW
        else:
            return None


def timer_wrapper(message, func, *tuple_args):
    if message is None:
        message = func.__name__.upper()
    six.print_("\033[95mStarting {}. \033[0m".format(message))
    start = time.time()
    # call function
    ret = func(*tuple_args)
    six.print_('\033[95m' + "{} consumes: {} s. \033[0m\n".format(message,
                                                                  round(time.time() - start, 2)))
    return ret
