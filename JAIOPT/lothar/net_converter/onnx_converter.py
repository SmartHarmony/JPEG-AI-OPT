import sys
from enum import Enum
import six

from deepvan.proto import deepvan_pb2
from lothar.net_converter import base_converter
from lothar.net_converter.base_converter import PoolingType
from lothar.net_converter.base_converter import PaddingMode
from lothar.net_converter.base_converter import PaddingType
from lothar.net_converter.base_converter import ActivationType
from lothar.net_converter.base_converter import EltwiseType
from lothar.net_converter.base_converter import ReduceType
from lothar.net_converter.base_converter import FrameworkType
from lothar.net_converter.base_converter import RoundMode
from lothar.net_converter.base_converter import DataFormat
from lothar.net_converter.base_converter import DeepvanOp
from lothar.net_converter.base_converter import DeepvanConfigKey
from lothar.net_converter.base_converter import ConverterUtil
from lothar.net_converter.base_converter import timer_wrapper
from lothar.net_converter.base_converter import DeviceType
from lothar.net_converter.convert_util import CONDITIONS

import numpy as np

import onnx
import onnx.utils
from onnx import mapping, numpy_helper, TensorProto
from numbers import Number

IS_PYTHON3 = sys.version_info > (3,)


class AttributeType(Enum):
    INT = 100
    FLOAT = 101
    INTS = 102
    FLOATS = 103
    BOOL = 104


OnnxSupportedOps = [
    'Abs',
    'Add',
    'Conv',
    'ConvTranspose',
    'Div',
    'LeakyRelu',
    'Mul',
    'Relu',
    'Sub',
    'Transpose',
]

OnnxOpType = Enum('OnnxOpType',
                  [(op, op) for op in OnnxSupportedOps],
                  type=str)


def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


def convert_onnx(attr):
    return convert_onnx_attribute_proto(attr)


def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8')\
            if IS_PYTHON3 else attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        if IS_PYTHON3:
            str_list = map(lambda x: str(x, 'utf-8'), str_list)
        return str_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


class OnnxNode(object):
    def __init__(self, node):
        self.name = str(node.name)
        if self.name == '':
            self.name = str(node.output)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name,
                            translate_onnx(attr.name, convert_onnx(attr)))
                           for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node

    def print_info(self):
        print("node: ", self.name)
        print("    type: ", self.op_type)
        print("    domain: ", self.domain)
        print("    inputs: ", self.inputs)
        print("    outputs: ", self.outputs)
        print("    attrs:")
        for arg in self.attrs:
            print("        %s: %s" % (arg, self.attrs[arg]))


class OnnxTensor(object):
    def __init__(self, name, value, shape, dtype):
        self._name = name
        self._tensor_data = value
        self._shape = shape
        self._dtype = dtype


class OnnxConstantFolder():
    def __init__(self, onnx_graph):
        self._op_folder = {
            "Constant": self.fold_constant_node,
            "Div": self.fold_general_node,
            "Mul": self.fold_general_node,
            "Transpose": self.fold_general_node,
        }

        self._graph = onnx_graph
        self._graph_shapes_dict = {}
        self._consts = []

    def construct_consts(self):
        initializer = self._graph.initializer
        del self._consts[:]
        self._consts = [init.name for init in initializer]

    @staticmethod
    def save_model(model, file_path):
        onnx.save_model(model, file_path)

    @staticmethod
    def print_tensor_proto(graph, node, index):
        CONDITIONS(len(node.input) > index, "Index cann't larger than size")
        tensor_name = node.input[index]
        tensor_protos = [
            tensor for tensor in graph.initializer if tensor.name == tensor_name]
        CONDITIONS(len(tensor_protos) > 0, "Cannot find tensor proto")
        numpy_tensor = numpy_helper.to_array(tensor_protos[0])
        print("tensor proto name: %s, value: %s"
              % (tensor_name, numpy_tensor))

    def run(self):
        OnnxConverter.extract_shape_info(self._graph_shapes_dict, self._graph)
        self.construct_consts()

        # inner function to fold operation/onnx_node
        def inner_fold():
            for node in filter(lambda n: self.is_constant_operation(n)
                               and n.op_type in self._op_folder, self._graph.node):
                result = self._op_folder[node.op_type](node)
                print("Safely remove Node: {}({}: {}->{})".format(node.op_type,
                      node.name, node.input, node.output))
                self.construct_consts()
                return result
            return False

        while True:
            changed = inner_fold()
            if not changed:
                break

        # for debug
        for node in self._graph.node:
            if node.op_type == "Upsample":
                # self.print_tensor_proto(self._graph, node, 1)
                pass

    def fold_general_node(self, node):
        inputs = {name: self.get_tensor_proto(
            self._graph, name) for name in node.input}
        output_name = node.output[0]
        result = self.run_node(node, inputs)
        self.add_tensor_proto(*self.parse_node_result(node, result))
        self.safe_remove_node(node)
        return True

    def fold_shape_node(self, node):
        """
        This fold function is important. Shape is the entry
        to fold some unuseful node.
        """
        CONDITIONS(len(node.input) == 1 and len(node.output) == 1,
                   "Cannot fold this shape %s" % node.name)
        if node.input[0] not in self._graph_shapes_dict:
            print("check fold shape of onnx")
            return False
        input_shape = self._graph_shapes_dict[node.input[0]]
        output_name = node.output[0]
        self.add_tensor_proto(output_name, TensorProto.INT64,
                              [len(input_shape)], [long(x) for x in input_shape])
        self.safe_remove_node(node)
        return True

    def fold_constant_node(self, node):
        attrs = dict([(attr.name,
                       translate_onnx(attr.name, convert_onnx(attr))) for attr in node.attribute])
        tensor_proto = attrs['value']
        tensor_proto.name = node.output[0]
        self._graph.initializer.extend([tensor_proto])
        self.safe_remove_node(node)
        return True

    def add_tensor_proto(self, tensor_name, dtype, dims, vals):
        tensor_proto = onnx.helper.make_tensor(
            name=tensor_name,
            data_type=dtype,
            dims=dims,
            vals=vals
        )
        self._consts.append(tensor_name)
        self._graph.initializer.extend([tensor_proto])

    def safe_remove_node(self, node):
        self._graph.node.remove(node)

    def get_node_name(self, node):
        if not node.name or node.name == '':
            return node.output[0]
        return node.name

    def is_constant_operation(self, node):
        if node.op_type == "Shape":
            return True
        return not any(name not in self._consts for name in node.input)

    @staticmethod
    def run_node(node, inputs=None, **kwargs):
        import caffe2.python.onnx.backend as caffe2_backend
        return caffe2_backend.run_node(node=node, inputs=inputs)

    def parse_node_result(self, node, outputs):
        CONDITIONS(len(node.output) == 1, "Output cannot be larger than 1")
        tensor_name = node.output[0]
        output_numpy = outputs[tensor_name]
        dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[output_numpy.dtype]
        return tensor_name, dtype, output_numpy.shape, output_numpy.flat

    @classmethod
    def get_tensor_proto(cls, graph, name, to_numpy=True):
        tensor_proto = next(
            x for x in graph.initializer if x.name == name)
        if not tensor_proto:
            raise Exception("Cannot get tensor proto %s" % name)
        return OnnxConstantFolder.tensor_proto_to_numpy(tensor_proto) if to_numpy else tensor_proto

    @classmethod
    def tensor_proto_to_numpy(cls, tensor_proto):
        numpy_tensor = numpy_helper.to_array(tensor_proto)
        return numpy_tensor


class OnnxConverter(base_converter.ConverterInterface):
    pooling_type_mode = {

    }

    auto_pad_mode = {
        'NOTSET': PaddingMode.NA,
        'SAME_UPPER': PaddingMode.SAME,
        'SAME_LOWER': PaddingMode.SAME,
        'VALID': PaddingMode.VALID,
    }
    auto_pad_mode = {six.b(k): v for k, v in six.iteritems(auto_pad_mode)}

    eltwise_type = {
        OnnxOpType.Mul.name: EltwiseType.PROD,
        OnnxOpType.Add.name: EltwiseType.SUM,
        OnnxOpType.Abs.name: EltwiseType.ABS,
        OnnxOpType.Sub.name: EltwiseType.SUB,
        OnnxOpType.Div.name: EltwiseType.DIV,
    }

    reduce_type = {

    }

    activation_type = {
        OnnxOpType.Relu.name: ActivationType.RELU,
        OnnxOpType.LeakyRelu.name: ActivationType.LEAKYRELU,
    }

    def __init__(self, option, src_model_file):
        self._op_converters = {
            OnnxOpType.Abs.name: self.convert_eltwise,
            OnnxOpType.Add.name: self.convert_eltwise,
            OnnxOpType.Conv.name: self.convert_conv2d,
            OnnxOpType.ConvTranspose.name: self.convert_deconv,
            OnnxOpType.Div.name: self.convert_eltwise,
            OnnxOpType.LeakyRelu.name: self.convert_activation,
            OnnxOpType.Mul.name: self.convert_eltwise,
            OnnxOpType.Relu.name: self.convert_activation,
            OnnxOpType.Sub.name: self.convert_eltwise,
            OnnxOpType.Transpose.name: self.convert_transpose,
        }
        self._option = option
        self._net_def = deepvan_pb2.NetProto()
        self._data_format = DataFormat.NCHW
        ConverterUtil.set_filter_format(self._net_def, DataFormat.OIHW)
        ConverterUtil.add_data_format_arg(self._net_def,
                                          self._data_format)
        onnx_model = onnx.load(src_model_file)

        self.ir_version = onnx_model.ir_version
        opset_imp = onnx_model.opset_import

        try:
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print(e)

        self._isKaldi = False

        polish_available = False
        print("onnx model IR version: ", self.ir_version)
        for imp in opset_imp:
            domain = imp.domain
            self.op_version = imp.version
            print("constains ops domain: ", domain,
                  "version:", self.op_version)
            if 'kaldi' in domain:
                polish_available = False
                self._data_format = DataFormat.NONE
                self._isKaldi = True
        if polish_available:
            onnx_model = onnx.utils.polish_model(onnx_model)

        self._onnx_model = onnx_model
        self._graph_shapes_dict = {}
        self._consts = {}
        self._replace_tensors = {}

        self._constant_folder = OnnxConstantFolder(self._onnx_model.graph)

    @staticmethod
    def print_graph_info(graph):
        for value_info in graph.value_info:
            print("value info:", value_info)
        for value_info in graph.input:
            print("inputs info:", value_info)
        for value_info in graph.output:
            print("outputs info:", value_info)

    @staticmethod
    def extract_shape_info(graph_shapes_dict, graph):
        def extract_value_info(shape_dict, value_info):
            t = tuple([int(dim.dim_value)
                      for dim in value_info.type.tensor_type.shape.dim])
            if t:
                shape_dict[value_info.name] = t

        for vi in graph.value_info:
            extract_value_info(graph_shapes_dict, vi)
        for vi in graph.input:
            extract_value_info(graph_shapes_dict, vi)
        for vi in graph.output:
            extract_value_info(graph_shapes_dict, vi)
        # also need to all initializer's shape
        for init in graph.initializer:
            graph_shapes_dict[init.name] = list(init.dims)

    def add_tensor(self, name, shape, data_type, value, need_flat=True):
        tensor = self._net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        if need_flat:
            tensor.float_data.extend(value.flat)
        else:
            tensor.float_data.extend(value)

    def run(self):
        # remove pre-calculated node
        timer_wrapper(None, self._constant_folder.run)
        # self._constant_folder.save_model(self._onnx_model, "/xxx.onnx")
        # infer shape into value_info in ModelGraph
        try:
            self._onnx_model = timer_wrapper(
                None, onnx.shape_inference.infer_shapes, self._onnx_model)
        except:
            # import ipdb; ipdb.set_trace()
            pass
        graph_def = self._onnx_model.graph
        timer_wrapper(None, self.extract_shape_info,
                      self._graph_shapes_dict, graph_def)
        timer_wrapper(None, self.convert_tensors, graph_def)
        timer_wrapper(None, self.convert_ops, graph_def)
        return self._net_def

    def add_stride_pad_kernel_arg(self, attrs, op_def):
        if 'strides' in attrs:
            strides = attrs['strides']
            CONDITIONS(len(strides) >= 2, "strides should has 2 values.")
            # stride = [strides[0], strides[1]]
            stride = list(strides)
        else:
            stride = [1, 1]

        strides_arg = op_def.arg.add()
        strides_arg.name = DeepvanConfigKey.deepvan_strides_str
        strides_arg.ints.extend(stride)

        if 'kernel_shape' in attrs:
            kernel_shape = attrs['kernel_shape']
            CONDITIONS(len(kernel_shape) >= 2,
                       "kernel shape should has 2 values.")
            # kernel = [kernel_shape[0], kernel_shape[1]]
            kernel = list(kernel_shape)
            kernels_arg = op_def.arg.add()
            kernels_arg.name = DeepvanConfigKey.deepvan_kernel_str
            kernels_arg.ints.extend(kernel)

        # TODO: Does not support AutoPad yet.
        if 'pads' in attrs:
            pads = attrs['pads']
            if len(pads) == 4:
                pad = [pads[0] + pads[2], pads[1] + pads[3]]
            elif len(pads) == 6:
                pad = [pads[0] + pads[3], pads[1] + pads[4], pads[2] + pads[5]]
            else:
                pad = [0] * (len(op_def.output_shape[0].dims) - 2)
            padding_arg = op_def.arg.add()
            padding_arg.name = DeepvanConfigKey.deepvan_padding_values_str
            padding_arg.ints.extend(pad)
        elif 'auto_pad' in attrs:  # deprecateed
            auto_pad_arg = op_def.arg.add()
            auto_pad_arg.name = DeepvanConfigKey.deepvan_padding_str
            auto_pad_arg.i = self.auto_pad_mode[attrs['auto_pad']].value
        else:
            pad = [0] * (len(op_def.output_shape[0].dims) - 2)
            padding_arg = op_def.arg.add()
            padding_arg.name = DeepvanConfigKey.deepvan_padding_values_str
            padding_arg.ints.extend(pad)

    def remove_node(self, node):
        input_name = node.inputs[0]
        output_name = node.outputs[0]
        self._replace_tensors[output_name] = input_name

    @staticmethod
    def squeeze_shape(shape, axis):
        new_shape = []
        if len(axis) > 0:
            for i in range(len(shape)):
                if i not in axis:
                    new_shape.append(shape[i])
        else:
            new_shape = shape
        return new_shape

    @staticmethod
    def unsqueeze_shape(shape, axis):
        new_shape = [n for n in shape]
        for n in axis:
            new_shape.insert(n, 1)
        return new_shape

    @staticmethod
    def transpose_const(tensor):
        shape = tensor.dims
        CONDITIONS(len(shape) == 2, "gemm only supports 2-dim input.")
        tensor_data = np.array(tensor.float_data).reshape(
            shape[0], shape[1])
        tensor_data = tensor_data.transpose(1, 0)
        tensor.float_data[:] = tensor_data.flat
        tensor.dims[:] = tensor_data.shape

    def convert_ops(self, graph_def):
        for n in graph_def.node:
            node = OnnxNode(n)
            CONDITIONS(node.op_type in self._op_converters,
                       "[UNSUPPORTED] \"%s\" is not supported according to its OP type." % node.op_type)

            self._op_converters[node.op_type](node)

    def convert_tensors(self, graph_def):
        initializer = graph_def.initializer
        if initializer:
            for init in initializer:
                tensor = self._net_def.tensors.add()
                tensor.name = init.name

                onnx_tensor = numpy_helper.to_array(init)
                tensor.dims.extend(list(init.dims))
                data_type = onnx_dtype(init.data_type)

                if data_type == np.float32 or data_type == np.float64:
                    tensor.data_type = deepvan_pb2.DT_FLOAT
                    tensor.float_data.extend(
                        onnx_tensor.astype(np.float32).flat)
                elif data_type == np.int32:
                    tensor.data_type = deepvan_pb2.DT_INT32
                    tensor.int32_data.extend(
                        onnx_tensor.astype(np.int32).flat)
                elif data_type == np.int64:
                    tensor.data_type = deepvan_pb2.DT_INT32
                    tensor.int32_data.extend(
                        onnx_tensor.astype(np.int32).flat)
                elif data_type == np.bool:
                    tensor.data_type = deepvan_pb2.DT_INT32
                    tensor.int32_data.extend(
                        onnx_tensor.astype(np.int32).flat)
                else:
                    CONDITIONS(False,
                               "[UNSUPPORTED] \"%s\" is not supported as tensor type for (%s)" % (data_type, init.name))
                self._consts[tensor.name] = tensor

    def convert_general_op(self, node, with_shape=True):
        op = self._net_def.op.add()
        op.name = node.name

        for input in node.inputs:
            if input in self._replace_tensors:
                input = self._replace_tensors[input]
            op.input.append(input)
        for output in node.outputs:
            op.output.append(output)
            if with_shape:
                if output in self._graph_shapes_dict:
                    output_shape = op.output_shape.add()
                    shape_info = self._graph_shapes_dict[output]
                    output_shape.dims.extend(shape_info)

        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = DeepvanConfigKey.deepvan_framework_type_str
        framework_type_arg.i = FrameworkType.ONNX.value

        ConverterUtil.add_data_format_arg(op, self._data_format)
        return op

    def convert_activation(self, node):
        op = self.convert_general_op(node)
        op.type = DeepvanOp.Activation.name

        type_arg = op.arg.add()
        type_arg.name = DeepvanConfigKey.deepvan_activation_type_str
        type_arg.s = six.b(self.activation_type[node.op_type].name)

        if "alpha" in node.attrs:
            alpha_value = node.attrs["alpha"]
        else:
            if node.op_type == OnnxOpType.LeakyRelu.name:
                alpha_value = 0.01
            else:
                alpha_value = 0
        alpha_arg = op.arg.add()
        if node.op_type == OnnxOpType.LeakyRelu.name:
            alpha_arg.name = DeepvanConfigKey.deepvan_activation_leakyrelu_coefficient_str
        else:
            alpha_arg.name = DeepvanConfigKey.deepvan_activation_max_limit_str
        alpha_arg.f = alpha_value

    def convert_conv2d(self, node):
        if len(self._graph_shapes_dict[node.outputs[0]]) == 5:
            return self.convert_conv3d(node)
        op = self.convert_general_op(node)
        self.add_stride_pad_kernel_arg(node.attrs, op)
        group_arg = op.arg.add()
        group_arg.name = DeepvanConfigKey.deepvan_group_str
        if 'group' in node.attrs:
            group_val = node.attrs["group"]
        else:
            group_val = 1
        group_arg.i = group_val

        is_depthwise = False
        if group_val > 1:
            filter_shape = self._graph_shapes_dict[node.inputs[1]]
            CONDITIONS(group_val == filter_shape[0] and
                       filter_shape[1] == 1,
                       "[UNSUPPORTED] \"%s\" with group convolution is not supported" % node.op_type)
            filter_tensor = self._consts[node.inputs[1]]
            new_shape = [filter_shape[1], filter_shape[0],
                         filter_shape[2], filter_shape[3]]
            del filter_tensor.dims[:]
            filter_tensor.dims.extend(new_shape)
            is_depthwise = True
        if is_depthwise:
            op.type = DeepvanOp.DepthwiseConv2d.name
        else:
            op.type = DeepvanOp.Conv2D.name
            CONDITIONS(op.input[1] in self._consts,
                       "[UNSUPPORTED] \"%s\" with non-const filter convolution is not supported" % node.op_type)

        dilation_arg = op.arg.add()
        dilation_arg.name = DeepvanConfigKey.deepvan_dilations_str
        if 'dilations' in node.attrs:
            dilation_val = node.attrs["dilations"]
        else:
            dilation_val = [1, 1]
        dilation_arg.ints.extend(dilation_val)

        if node.op_type == 'ConvSwish':
            activation_arg = op.arg.add()
            activation_arg.name = DeepvanConfigKey.deepvan_activation_type_str
            activation_arg.s = six.b("SWISH")
        else:
            if node.attrs.get('activation_type'):
                activation_arg = op.arg.add()
                activation_arg.name = DeepvanConfigKey.deepvan_activation_type_str
                activation_arg.s = six.b(node.attrs.get('activation_type'))

    def convert_deconv(self, node):
        op = self.convert_general_op(node)

        self.add_stride_pad_kernel_arg(node.attrs, op)

        if 'group' in node.attrs:
            group_val = node.attrs["group"]
        else:
            group_val = 1
        if group_val > 1:
            op.type = DeepvanOp.DepthwiseDeconv2d.name
            filter_shape = self._graph_shapes_dict[node.inputs[1]]
            filter_tensor = self._consts[node.inputs[1]]
            new_shape = [filter_shape[1], filter_shape[0],
                         filter_shape[2], filter_shape[3]]
            del filter_tensor.dims[:]
            filter_tensor.dims.extend(new_shape)
        else:
            op.type = DeepvanOp.Deconv2D.name
        group_arg = op.arg.add()
        group_arg.name = DeepvanConfigKey.deepvan_group_str
        group_arg.i = group_val

        dilation_arg = op.arg.add()
        dilation_arg.name = DeepvanConfigKey.deepvan_dilations_str
        if 'dilations' in node.attrs:
            dilation_val = node.attrs["dilations"]
        else:
            dilation_val = [1, 1]
        dilation_arg.ints.extend(dilation_val)
        CONDITIONS(dilation_val == [1, 1],
                   "[UNSUPPORTED] \"%s\" (convtranspose) with dilation != 1 is not supported" % node.op_type)

        # CONDITIONS('output_padding' not in node.attrs,
        #            "[UNSUPPORTED] \"%s\" (convtranspose) with output_padding is not supported" % node.op_type)
        CONDITIONS('output_shape' not in node.attrs,
                   "[UNSUPPORTED] \"%s\" (convtranspose) with output_shape is not supported" % node.op_type)
        # TODO: if output shape specified, calculate padding value
        if 'output_padding' in node.attrs:
            output_padding = node.attrs['output_padding']
            output_padding_arg = op.arg.add()
            output_padding_arg.name = DeepvanConfigKey.deepvan_output_padding_str
            output_padding_arg.ints.extend(output_padding)
        # if 'output_shape' in node.attrs:
        #     output_shape = node.attrs['output_shape']
        #     output_shape_arg = op.arg.add()
        #     output_shape_arg.name = DeepvanConfigKey.deepvan_output_shape_str
        #     output_shape_arg.ints.extend(output_shape)

    def convert_eltwise(self, node):
        op = self.convert_general_op(node)
        # todo: be aware it may cause a bug

        op.type = DeepvanOp.Eltwise.name
        type_arg = op.arg.add()
        type_arg.name = DeepvanConfigKey.deepvan_element_type_str
        type_arg.i = self.eltwise_type[node.op_type].value

        if node.op_type in [OnnxOpType.Div.name, OnnxOpType.Add.name, OnnxOpType.Sub.name]:
            inputs = list(node.inputs)
            if len(inputs) == 2 and (inputs[1] not in self._graph_shapes_dict
                                     or len(self._graph_shapes_dict[inputs[1]]) == 0):
                scalar_tensor = OnnxConstantFolder.get_tensor_proto(
                    self._onnx_model.graph, inputs[1])
                CONDITIONS(len(scalar_tensor.shape) == 0, "Error")
                scalar_value = list(scalar_tensor.flat)[0]
                # TODO: change node/op type
                # convert div to mul
                # if scalar_value != 0:
                #     type_arg.i = self.eltwise_type["Mul"].value
                #     scalar_value = 1.0 / scalar_value
                # add scalar input
                value_arg = op.arg.add()
                value_arg.name = DeepvanConfigKey.deepvan_scalar_input_str
                value_arg.f = scalar_value
                # remove input
                origin_input = inputs[1]
                inputs.remove(origin_input)
                del op.input[:]
                op.input.extend(inputs)

    @staticmethod
    def copy_node_attr(op, node, attr_name, dtype=AttributeType.INT,
                       default=None):
        if attr_name in node.attrs or default is not None:
            if attr_name in node.attrs:
                value = node.attrs[attr_name]
            else:
                value = default
            new_arg = op.arg.add()
            new_arg.name = attr_name
            if dtype == AttributeType.INT:
                new_arg.i = int(value)
            elif dtype == AttributeType.FLOAT:
                new_arg.f = float(value)
            elif dtype == AttributeType.INTS:
                new_arg.ints.extend(value)
            elif dtype == AttributeType.FLOATS:
                new_arg.floats.extend(value)
            return value
        else:
            return default

    def convert_transpose(self, node):
        op = self.convert_general_op(node)
        op.type = DeepvanOp.Transpose.name

        if 'perm' in node.attrs:
            perm = node.attrs['perm']
            ordered_perm = np.sort(perm)
            if np.array_equal(perm, ordered_perm):
                op.type = DeepvanOp.Identity.name
                del op.input[1:]
            else:
                dims_arg = op.arg.add()
                dims_arg.name = DeepvanConfigKey.deepvan_dims_str
                dims_arg.ints.extend(perm)
