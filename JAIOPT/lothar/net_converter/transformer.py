import re

import numpy as np
import six
import time
import sys
from termcolor import colored

from deepvan.proto import deepvan_pb2
from lothar.net_converter import base_converter
from lothar.net_converter.base_converter import ConverterUtil
from lothar.net_converter.base_converter import DataFormat
from lothar.net_converter.base_converter import DeviceType
from lothar.net_converter.base_converter import DeepvanConfigKey
from lothar.net_converter.base_converter import DeepvanOp
from lothar.net_converter.base_converter import DeepvanFixedDataFormatOps  # noqa
from lothar.net_converter.base_converter import DeepvanTransposableDataFormatOps  # noqa
from lothar.net_converter.base_converter import TransformerRule
from lothar.net_converter.base_converter import timer_wrapper
from lothar.net_converter.convert_util import CONDITIONS


class Transformer(base_converter.ConverterInterface):
    """A class for transform naive deepvan model to optimized model.
    This Transformer should be platform irrelevant. So, do not assume
    tensor name has suffix like ':0".
    """

    def __init__(self, option, model):
        # Dependencies
        # (TRANSFORM_MATMUL_TO_FC, TRANSFORM_GLOBAL_CONV_TO_FC) -> RESHAPE_FC_WEIGHT  # noqa
        self._registered_transformers = {
            TransformerRule.TRANSPOSE_FILTERS: self.transpose_filters,
            TransformerRule.ADD_IN_OUT_TENSOR_INFO:
                self.add_in_out_tensor_info,
            TransformerRule.ADD_WINOGRAD_ARG: self.add_winograd_arg,
            TransformerRule.ADD_OPENCL_INFORMATIONS:
                self.add_opencl_informations,
            TransformerRule.SORT_BY_EXECUTION: self.sort_by_execution,
            TransformerRule.UPDATE_DATA_FORMAT: self.update_data_format,
            TransformerRule.TRANSPOSE_DATA_FORMAT: self.transpose_data_format,
        }

        self._option = option
        self._model = model
        self._wino_arg = self._option.winograd

        self._ops = {}
        self._consts = {}
        self._consumers = {}
        self._producer = {}

        self._pattern_config = {}
        self._pattern_gap = {}

        self.input_name_map = {}
        self.output_name_map = {}
        self.initialize_name_map()

        # for conv folded bug
        self.folded_filter = []
        self.folded_conv_bias = []

    def run(self):

        for key in self._option.transformer_option:
            def inner_transformer():
                transformer = self._registered_transformers[key]
                while True:
                    self.construct_ops_and_consumers(key)
                    changed = transformer()
                    if not changed:
                        break
            func_name = self._registered_transformers[key].__name__
            timer_wrapper(func_name, inner_transformer)
        while True:
            change = self.transform_transpose_reshape()
            if not change:
                break

        return self._model, {}

    def initialize_name_map(self):
        for input_node in self._option.input_nodes.values():
            new_input_name = DeepvanConfigKey.deepvan_input_node_name \
                + '_' + input_node.name
            self.input_name_map[input_node.name] = new_input_name

        output_nodes = self._option.check_nodes.values()
        for output_node in output_nodes:
            new_output_name = DeepvanConfigKey.deepvan_output_node_name \
                + '_' + output_node.name
            self.output_name_map[output_node.name] = new_output_name

    def filter_format(self):
        filter_format_value = ConverterUtil.get_arg(self._model,
                                                    DeepvanConfigKey.deepvan_filter_format_str).i  # noqa
        filter_format = None
        if filter_format_value == DataFormat.HWIO.value:
            filter_format = DataFormat.HWIO
        elif filter_format_value == DataFormat.OIHW.value:
            filter_format = DataFormat.OIHW
        elif filter_format_value == DataFormat.HWOI.value:
            filter_format = DataFormat.HWOI
        else:
            CONDITIONS(False, "filter format %d not supported" %
                       filter_format_value)
        return filter_format

    def construct_ops_and_consumers(self, key):
        self._ops.clear()
        self._consumers.clear()
        self._producer.clear()
        for op in self._model.op:
            self._ops[op.name] = op
        for tensor in self._model.tensors:
            self._consts[tensor.name] = tensor
        for op in self._ops.values():
            for input_tensor in op.input:
                if input_tensor not in self._consumers:
                    self._consumers[input_tensor] = []
                self._consumers[input_tensor].append(op)

            for output_tensor in op.output:
                self._producer[output_tensor] = op
        if key != TransformerRule.SORT_BY_EXECUTION:
            for input_node in self._option.input_nodes.values():
                input_node_existed = False
                for op in self._model.op:
                    if input_node.name in op.output:
                        input_node_existed = True
                        break
                if not input_node_existed:
                    op = deepvan_pb2.OperatorProto()
                    op.name = self.normalize_op_name(input_node.name)
                    op.type = "Input"
                    data_type_arg = op.arg.add()
                    data_type_arg.name = DeepvanConfigKey.deepvan_op_data_type_str
                    data_type_arg.i = input_node.data_type
                    op.output.extend([input_node.name])
                    output_shape = op.output_shape.add()
                    output_shape.dims.extend(input_node.shape)
                    if input_node.data_format != DataFormat.DF_NONE:
                        if input_node.data_format == DataFormat.NCHW:
                            self.transpose_shape(output_shape.dims,
                                                 [0, 3, 1, 2])
                        ConverterUtil.add_data_format_arg(op,
                                                          DataFormat.AUTO)
                    else:
                        ConverterUtil.add_data_format_arg(op,
                                                          DataFormat.DF_NONE)
                    self._producer[op.output[0]] = op

    @staticmethod
    def replace(obj_list, source, target):
        for i in six.moves.range(len(obj_list)):
            if obj_list[i] == source:
                obj_list[i] = target

    @staticmethod
    def transpose_shape(shape, order):
        transposed_shape = []
        for i in six.moves.range(len(order)):
            transposed_shape.append(shape[order[i]])
        shape[:] = transposed_shape[:]

    @staticmethod
    def normalize_op_name(name):
        return name.replace(':', '_')

    def consumer_count(self, tensor_name):
        return len(self._consumers.get(tensor_name, []))

    def safe_remove_node(self, op, replace_op, remove_input_tensor=False):
        """remove op.
        1. change the inputs of its consumers to the outputs of replace_op
        2. if the op is output node, change output node to replace op"""

        if replace_op is None:
            # When no replace op specified, we change the inputs of
            # its consumers to the input of the op. This handles the case
            # that the op is identity op and its input is a tensor.
            reshape_const_dim = op.type == DeepvanOp.Reshape.name and \
                (len(op.input) == 1 or op.input[1] in self._consts)

            CONDITIONS(len(op.output) == 1 and len(op.input) == 1 or reshape_const_dim,
                       "cannot remove op that w/o replace op specified and input/output length > 1\n" + str(op))

            for consumer_op in self._consumers.get(op.output[0], []):
                self.replace(consumer_op.input, op.output[0], op.input[0])

            CONDITIONS(op.output[0] not in self._option.output_nodes,
                       "cannot remove op that is output node")
        else:
            CONDITIONS(len(op.output) == len(replace_op.output),
                       "cannot remove op since len(op.output) != len(replace_op.output)")

            for i in six.moves.range(len(op.output)):
                for consumer_op in self._consumers.get(op.output[i], []):
                    self.replace(consumer_op.input,
                                 op.output[i],
                                 replace_op.output[i])

            # if the op is output node, change replace_op output name to the op
            # output name
            for i in six.moves.range(len(op.output)):
                if op.output[i] in self._option.output_nodes:
                    for consumer in self._consumers.get(
                            replace_op.output[i], []):
                        self.replace(consumer.input,
                                     replace_op.output[i],
                                     op.output[i])
                    replace_op.output[i] = op.output[i]

        if remove_input_tensor:
            for input_name in op.input:
                if input_name in self._consts:
                    const_tensor = self._consts[input_name]
                    self._model.tensors.remove(const_tensor)

        self._model.op.remove(op)

    def add_in_out_tensor_info(self):
        net = self._model
        for input_node in self._option.input_nodes.values():
            input_info = net.input_info.add()
            input_info.name = input_node.name
            input_info.data_format = input_node.data_format.value
            input_info.dims.extend(input_node.shape)
            input_info.data_type = input_node.data_type

        output_nodes = self._option.check_nodes.values()
        for output_node in output_nodes:
            output_info = net.output_info.add()
            output_info.name = output_node.name
            output_info.data_format = output_node.data_format.value
            # TODO @vgod
            # output_info.dims.extend(
            #     self._producer[output_node.name].output_shape[0].dims)
            output_info.dims.extend(output_node.shape)
            output_info.data_type = output_node.data_type

        return False

    @staticmethod
    def sort_feature_map_shape(shape, data_format):
        """Return shape in NHWC order"""
        batch = shape[0]
        if data_format == DataFormat.NHWC:
            height = shape[1]
            width = shape[2]
            channels = shape[3]
        else:
            height = shape[2]
            width = shape[3]
            channels = shape[1]
        return batch, height, width, channels

    @staticmethod
    def sort_filter_shape(filter_shape, filter_format):
        """Return filter shape in HWIO order"""
        if filter_format == DataFormat.HWIO:
            filter_height = filter_shape[0]
            filter_width = filter_shape[1]
            in_channels = filter_shape[2]
            out_channels = filter_shape[3]
        elif filter_format == DataFormat.OIHW:
            filter_height = filter_shape[2]
            filter_width = filter_shape[3]
            in_channels = filter_shape[1]
            out_channels = filter_shape[0]
        elif filter_format == DataFormat.HWOI:
            filter_height = filter_shape[0]
            filter_width = filter_shape[1]
            in_channels = filter_shape[3]
            out_channels = filter_shape[2]
        else:
            CONDITIONS(False, "filter format %s not supported" % filter_format)
        return filter_height, filter_width, in_channels, out_channels

    def transform_transpose_reshape(self):
        if self._option.device != DeviceType.GPU.value \
                or self._option.cl_mem_type != "image":
            return False
        net = self._model
        transpose_reshape_key = "transpose_reshape"
        for op in net.op:
            if (op.type == DeepvanOp.Transpose.name and ConverterUtil.get_arg(op,
                                                                              transpose_reshape_key) == None):
                output_shape = op.output_shape[0].dims
                for arg in op.arg:
                    if arg.name == "dims":
                        dims = arg.ints
                if (len(output_shape) == 3 and (output_shape[0] == 1 or output_shape[1] == 1) and dims == [1, 0, 2]):
                    print("Add transpose reshape 2 to: %s(%s)",
                          (op.name, op.type))
                    transpose_arg = op.arg.add()
                    transpose_arg.name = transpose_reshape_key
                    transpose_arg.i = 2  # reuse this argument
                    continue
            if (op.type == DeepvanOp.Reshape.name and ConverterUtil.get_arg(op,
                                                                            transpose_reshape_key) == None):
                if not self._consumers.get(op.output[0]):
                    continue
                consumer_op = self._consumers[op.output[0]][0]
                if (consumer_op.type == DeepvanOp.Transpose.name):
                    print("Add transpose reshape 1 to: %s(%s) and %s(%s)"
                          % (op.name, op.type, consumer_op.name, consumer_op.type))
                    transpose_arg = consumer_op.arg.add()
                    transpose_arg.name = transpose_reshape_key
                    transpose_arg.i = 1
                    reshape_arg = op.arg.add()
                    reshape_arg.name = transpose_reshape_key
                    reshape_arg.i = 1
                    return True

    def add_winograd_arg(self):
        if self._wino_arg == 0:
            return False
        net = self._model

        executing_only = self._option.executing_devices
        forbid_wino = executing_only.get('FORBID_WINO', [])
        for op in net.op:
            if op.type == DeepvanOp.Conv2D.name:
                winograd_arg = op.arg.add()
                winograd_arg.name = DeepvanConfigKey.deepvan_wino_arg_str
                winograd_arg.i = 0 if op.name.encode('utf-8') in forbid_wino \
                    else self._wino_arg

        return False

    def transpose_filters(self):
        net = self._model
        filter_format = self.filter_format()
        transposed_filter = set()
        transposed_deconv_filter = set()

        if self._option.quantize and \
                (self._option.device == DeviceType.CPU.value or
                 self._option.device == DeviceType.APU.value):
            print("Transpose filters to OHWI")
            if filter_format == DataFormat.HWIO:
                transpose_order = [3, 0, 1, 2]
            elif filter_format == DataFormat.OIHW:
                transpose_order = [0, 2, 3, 1]
            else:
                CONDITIONS(False, "Quantize model does not support conv "
                           "filter format: %s" % filter_format.name)

            for op in net.op:
                if (op.type == DeepvanOp.Conv2D.name or
                    op.type == DeepvanOp.Deconv2D.name) and\
                        op.input[1] not in transposed_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(transpose_order)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_filter.add(op.input[1])
            # deconv's filter's output channel and input channel is reversed
            for op in net.op:
                if op.type == DeepvanOp.Deconv2D.name and \
                        op.input[1] not in transposed_deconv_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(3, 1, 2, 0)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_deconv_filter.add(op.input[1])

            self.set_filter_format(DataFormat.OHWI)
        elif self._option.quantize and \
                (self._option.device == DeviceType.HEXAGON.value or
                 self._option.device == DeviceType.HTA.value):
            print("Transpose filters to HWIO/HWIM")
            CONDITIONS(filter_format == DataFormat.HWIO,
                       "HEXAGON only support HWIO/HWIM filter format.")
        else:
            # transpose filter to OIHW/MIHW for tensorflow (HWIO/HWIM)
            if filter_format == DataFormat.HWIO:
                for op in net.op:
                    if (op.type == DeepvanOp.Conv2D.name
                            or op.type == DeepvanOp.Deconv2D.name
                            or op.type == DeepvanOp.DepthwiseConv2d.name) \
                            and op.input[1] in self._consts \
                            and op.input[1] not in transposed_filter:
                        print("Transpose Conv2D/Deconv2D filters to OIHW/MIHW")
                        filter = self._consts[op.input[1]]
                        filter_data = np.array(filter.float_data).reshape(
                            filter.dims)
                        filter_data = filter_data.transpose(3, 2, 0, 1)
                        filter.float_data[:] = filter_data.flat
                        filter.dims[:] = filter_data.shape
                        transposed_filter.add(op.input[1])
                    if ((op.type == DeepvanOp.MatMul.name or op.type == DeepvanOp.Gemm.name) and
                            (ConverterUtil.get_arg(
                                op,
                                DeepvanConfigKey.deepvan_winograd_filter_transformed)
                             is not None)
                            and op.input[1] not in transposed_filter):
                        print("Transpose Winograd filters to OIHW/MIHW")
                        filter = self._consts[op.input[0]]
                        filter_data = np.array(filter.float_data).reshape(
                            filter.dims)
                        filter_data = filter_data.transpose(3, 2, 0, 1)
                        filter.float_data[:] = filter_data.flat
                        filter.dims[:] = filter_data.shape
                        transposed_filter.add(op.input[0])
                    if op.type == DeepvanOp.FullyConnected.name \
                            and op.input[1] not in transposed_filter:
                        weight = self._consts[op.input[1]]
                        if len(weight.dims) == 4:
                            print("Transpose FullyConnected filters to"
                                  " OIHW/MIHW")
                            weight_data = np.array(weight.float_data).reshape(
                                weight.dims)
                            weight_data = weight_data.transpose(3, 2, 0, 1)
                            weight.float_data[:] = weight_data.flat
                            weight.dims[:] = weight_data.shape
                            transposed_filter.add(op.input[1])

                self.set_filter_format(DataFormat.OIHW)
            # deconv's filter's output channel and input channel is reversed
            for op in net.op:
                if op.type in [DeepvanOp.Deconv2D.name,
                               #    DeepvanOp.DepthwiseDeconv2d
                               ] \
                        and op.input[1] not in transposed_deconv_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(1, 0, 2, 3)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_deconv_filter.add(op.input[1])

        return False

    def sort_dfs(self, op, visited, sorted_nodes):
        if op.name in visited:
            return
        visited.update([op.name])
        if len(op.input) > 0:
            for input_tensor in op.input:
                producer_op = self._producer.get(input_tensor, None)
                if producer_op is None:
                    pass
                elif producer_op.name not in visited:
                    self.sort_dfs(producer_op, visited, sorted_nodes)
        sorted_nodes.append(op)

    def sort_by_execution(self):
        print("Sort by execution")
        net = self._model
        visited = set()
        sorted_nodes = []

        output_nodes = self._option.check_nodes
        output_nodes.update(self._option.output_nodes)
        for output_node in output_nodes:
            CONDITIONS(output_node in self._producer,
                       "output_tensor %s not existed in model" % output_node)
            self.sort_dfs(self._producer[output_node], visited, sorted_nodes)

        del net.op[:]
        net.op.extend(sorted_nodes)

        print("Final ops:")
        index = 0
        for op in net.op:
            # if op.type not in [DeepvanOp.Quantize.name, DeepvanOp.Dequantize.name]:
            #     index_str = str(index)
            #     index += 1
            # else:
            index_str = ''
            print("%s (%s, index:%s): %s" % (op.name, op.type, index_str,
                  [out_shape.dims for out_shape in op.output_shape]))
        return False

    def update_data_format(self):
        print("update data format")
        net = self._model
        for op in net.op:
            df_arg = ConverterUtil.get_arg(
                op, DeepvanConfigKey.deepvan_data_format_str)
            if not df_arg:
                df_arg = op.arg.add()
                df_arg.name = DeepvanConfigKey.deepvan_data_format_str
            if op.type in DeepvanFixedDataFormatOps:
                df_arg.i = DataFormat.AUTO.value
            elif op.type in DeepvanTransposableDataFormatOps:
                input_df = DataFormat.AUTO.value
                for input_tensor in op.input:
                    if input_tensor in self._consts:
                        continue
                    if not input_tensor:
                        op.input.remove(input_tensor)
                        continue
                    CONDITIONS(
                        input_tensor in self._producer,
                        "Input tensor %s not in producer" % input_tensor)
                    father_op = self._producer[input_tensor]
                    temp_input_df = ConverterUtil.get_arg(
                        father_op, DeepvanConfigKey.deepvan_data_format_str)
                    if temp_input_df.i != DataFormat.AUTO.value:
                        input_df = temp_input_df.i
                if input_df == DataFormat.AUTO.value:
                    df_arg.i = input_df
                    # add flag to mark the ops may has data format
                    has_data_format_arg = op.arg.add()
                    has_data_format_arg.name = \
                        DeepvanConfigKey.deepvan_has_data_format_str
                    has_data_format_arg.i = 1
        return False

    def transpose_data_format(self):
        print("Transpose arguments based on data format")
        net = self._model

        src_data_format = ConverterUtil.data_format(net)
        for op in net.op:
            has_data_format = ConverterUtil.data_format(op) == \
                DataFormat.AUTO

            # transpose op output shape
            if src_data_format == DataFormat.NCHW and \
                    has_data_format:
                print("Transpose output shapes: %s(%s)" % (op.name, op.type))
                for output_shape in op.output_shape:
                    if len(output_shape.dims) == 4:
                        self.transpose_shape(output_shape.dims,
                                             [0, 2, 3, 1])
            else:
                print("--Not Transpose output shapes: %s(%s)" %
                      (op.name, op.type))

        return False

    def add_opencl_informations(self):
        print("Add OpenCL informations")

        net = self._model

        arg = net.arg.add()
        arg.name = DeepvanConfigKey.deepvan_opencl_mem_type
        arg.i = deepvan_pb2.GPU_IMAGE if self._option.cl_mem_type == "image"\
            else deepvan_pb2.GPU_BUFFER
