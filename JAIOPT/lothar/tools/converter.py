import argparse
import sys
import hashlib
import os.path
import json
import six

try:
    dirname = "/".join(os.path.dirname(os.path.abspath(__file__)
                                       ).split("/")[0:-2])
    sys.path.insert(0, dirname)
except Exception as e:
    print("Change work directory failed")
    exit(1)
from deepvan.proto import deepvan_pb2
from lothar.tools.model_persistent import save_model
from lothar.net_converter import base_converter as cvt
from lothar.net_converter import transformer
from lothar.net_converter.base_converter import timer_wrapper

# ./bazel-bin/path/tf_converter --model_file quantized_test.pb \
#                                            --output quantized_test_dsp.pb \
#                                            --runtime dsp \
#                                            --input_dim input_node,1,28,28,3

FLAGS = None

device_type_map = {'cpu': cvt.DeviceType.CPU.value,
                   'gpu': cvt.DeviceType.GPU.value,
                   'dsp': cvt.DeviceType.HEXAGON.value,
                   'hta': cvt.DeviceType.HTA.value,
                   'cpu+gpu': cvt.DeviceType.CPU.value}

data_format_map = {
    'NONE': cvt.DataFormat.DF_NONE,
    'NHWC': cvt.DataFormat.NHWC,
    'NCHW': cvt.DataFormat.NCHW,
    'OIHW': cvt.DataFormat.OIHW,
}

data_type_map = {
    'float32': deepvan_pb2.DT_FLOAT,
    'int32': deepvan_pb2.DT_INT32,
}


def parse_data_type(data_type, device_type):
    if device_type == cvt.DeviceType.CPU.value or \
            device_type == cvt.DeviceType.GPU.value:
        if data_type == 'float32_float32':
            return deepvan_pb2.DT_FLOAT
        else:
            return deepvan_pb2.DT_HALF
    elif device_type == cvt.DeviceType.HEXAGON.value or \
            device_type == cvt.DeviceType.HTA.value:
        return deepvan_pb2.DT_FLOAT
    else:
        print("Invalid device type: " + str(device_type))


def split_shape(shape):
    if shape.strip() == "":
        return []
    else:
        return shape.split(',')


def parse_int_array_from_str(ints_str):
    return [int(i) for i in split_shape(ints_str)]


def transpose_shape(shape, dst_order):
    t_shape = [0] * len(shape)
    for i in range(len(shape)):
        t_shape[i] = shape[dst_order[i]]
    return t_shape


def get_last_net_def(im_path):
    if not FLAGS.load_from_im or not os.path.exists(im_path):
        return None
    net_def = deepvan_pb2.NetProto()
    with open(im_path, 'rb') as f:
        net_def.ParseFromString(f.read())
    print("Load intermediate model from %s" % im_path)
    return net_def


def main(unused_args):
    if not os.path.isfile(FLAGS.model_file):
        six.print_("Input graph file '" +
                   FLAGS.model_file +
                   "' does not exist!", file=sys.stderr)
        sys.exit(-1)

    if FLAGS.platform == 'caffe':
        if not os.path.isfile(FLAGS.weight_file):
            six.print_("Input weight file '" + FLAGS.weight_file +
                       "' does not exist!", file=sys.stderr)
            sys.exit(-1)

    if FLAGS.platform not in ['tensorflow', 'caffe', 'onnx']:
        six.print_("platform %s is not supported." % FLAGS.platform,
                   file=sys.stderr)
        sys.exit(-1)
    if FLAGS.runtime not in ['cpu', 'gpu', 'dsp', 'hta', 'cpu+gpu']:
        six.print_("runtime %s is not supported." % FLAGS.runtime,
                   file=sys.stderr)
        sys.exit(-1)

    option = cvt.ConverterOption()
    if FLAGS.graph_optimize_options:
        option.transformer_option = FLAGS.graph_optimize_options.split(',')
    option.winograd = FLAGS.winograd
    option.quantize = FLAGS.quantize
    option.quantize_range_file = FLAGS.quantize_range_file
    option.pattern_weight = FLAGS.pattern_weight
    option.pattern_config_path = FLAGS.pattern_config_path
    option.model_tag = FLAGS.model_tag
    option.pattern_style_count = FLAGS.pattern_style_count
    option.conv_unroll = FLAGS.conv_unroll
    option.change_concat_ranges = FLAGS.change_concat_ranges
    option.cl_mem_type = FLAGS.cl_mem_type
    option.device = device_type_map[FLAGS.runtime]
    option.data_type = parse_data_type(FLAGS.data_type, option.device)
    option.model_type = FLAGS.model_type
    option.pruning_type = FLAGS.pruning_type
    option.executing_devices = json.loads(FLAGS.executing_devices)
    # option.executing_devices = byteify(json.loads(FLAGS.executing_devices))

    input_node_names = FLAGS.input_node.split(',')
    input_data_types = FLAGS.input_data_types.split(',')
    input_node_shapes = FLAGS.input_shape.split(':')
    input_node_formats = FLAGS.input_data_formats.split(",")
    if FLAGS.input_range:
        input_node_ranges = FLAGS.input_range.split(':')
    else:
        input_node_ranges = []
    if len(input_node_names) != len(input_node_shapes):
        raise Exception('input node count and shape count do not match.')
    for i in six.moves.range(len(input_node_names)):
        input_node = cvt.NodeInfo()
        input_node.name = input_node_names[i]
        input_node.data_type = data_type_map[input_data_types[i]]
        input_node.data_format = data_format_map[input_node_formats[i]]
        input_node.shape = parse_int_array_from_str(input_node_shapes[i])
        if input_node.data_format == cvt.DataFormat.NCHW and\
                len(input_node.shape) == 4:
            input_node.shape = transpose_shape(input_node.shape, [0, 2, 3, 1])
            input_node.data_format = cvt.DataFormat.NHWC
        if len(input_node_ranges) > i:
            input_node.range = parse_float_array_from_str(input_node_ranges[i])
        option.add_input_node(input_node)
        print("input node: ", str(input_node))

    output_node_names = FLAGS.output_node.split(',')
    output_data_types = FLAGS.output_data_types.split(',')
    output_node_shapes = FLAGS.output_shape.split(':')
    output_node_formats = FLAGS.output_data_formats.split(",")
    if len(output_node_names) != len(output_node_shapes):
        raise Exception('output node count and shape count do not match.')
    for i in six.moves.range(len(output_node_names)):
        output_node = cvt.NodeInfo()
        output_node.name = output_node_names[i]
        output_node.data_type = data_type_map[output_data_types[i]]
        output_node.data_format = data_format_map[output_node_formats[i]]
        output_node.shape = parse_int_array_from_str(output_node_shapes[i])
        if output_node.data_format == cvt.DataFormat.NCHW and\
                len(output_node.shape) == 4:
            output_node.shape = transpose_shape(output_node.shape,
                                                [0, 2, 3, 1])
            output_node.data_format = cvt.DataFormat.NHWC
        option.add_output_node(output_node)
        print("output node: ", str(output_node))

    if FLAGS.check_node != '':
        check_node_names = FLAGS.check_node.split(',')
        check_node_shapes = FLAGS.check_shape.split(':')
        if len(check_node_names) != len(check_node_shapes):
            raise Exception('check node count and shape count do not match.')
        for i in six.moves.range(len(check_node_names)):
            check_node = cvt.NodeInfo()
            check_node.name = check_node_names[i]
            check_node.shape = parse_int_array_from_str(check_node_shapes[i])
            option.add_check_node(check_node)
    else:
        option.check_nodes = option.output_nodes

    option.build()

    im_path = "builds/im/%s.pb" % FLAGS.model_tag
    output_graph_def = timer_wrapper(None, get_last_net_def, im_path)
    if output_graph_def is None:
        print("Transform model to one that can better run on device")
        if FLAGS.platform == 'tensorflow':
            from lothar.net_converter import tensorflow_converter
            converter = tensorflow_converter.TensorflowConverter(
                option, FLAGS.model_file)
        elif FLAGS.platform == 'caffe':
            from lothar.net_converter import caffe_converter
            converter = caffe_converter.CaffeConverter(option,
                                                       FLAGS.model_file,
                                                       FLAGS.weight_file)
        elif FLAGS.platform == 'onnx':
            from lothar.net_converter import onnx_converter
            converter = onnx_converter.OnnxConverter(option, FLAGS.model_file)
        else:
            six.print_("Deepvan do not support platorm %s yet." % FLAGS.platform,
                       file=sys.stderr)
            exit(1)
        # convert model
        output_graph_def = converter.run()
        # save im model
        if FLAGS.load_from_im:
            with open(im_path, "wb") as f:
                f.write(output_graph_def.SerializeToString())
    # transform/optimize model
    graph_transformer = transformer.Transformer(option, output_graph_def)
    output_graph_def, quantize_activation_info = graph_transformer.run()

    if option.device in [cvt.DeviceType.HEXAGON.value,
                         cvt.DeviceType.HTA.value]:
        from lothar.net_converter import hexagon_converter
        converter = hexagon_converter.HexagonConverter(
            option, output_graph_def, quantize_activation_info)
        output_graph_def = converter.run()

    import time
    start = time.time()
    save_model(
        option, output_graph_def,
        FLAGS.template_dir, FLAGS.obfuscate, FLAGS.model_tag,
        FLAGS.output_dir,
        FLAGS.winograd,
        FLAGS.pruning_type)
    print("Save model consumes: {} s".format(time.time() - start))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="TensorFlow \'GraphDef\' file to load, "
             "Onnx model file .onnx to load, "
             "Caffe prototxt file to load.")
    parser.add_argument(
        "--weight_file", type=str, default="", help="Caffe data file to load.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="File to save the output graph to.")
    parser.add_argument(
        "--runtime", type=str, default="", help="Runtime: cpu/gpu/dsp")
    parser.add_argument(
        "--input_node",
        type=str,
        default="input_node",
        help="e.g., input_node")
    parser.add_argument(
        "--input_data_types",
        type=str,
        default="float32",
        help="e.g., float32|int32")
    parser.add_argument(
        "--input_data_formats",
        type=str,
        default="NHWC",
        help="e.g., NHWC,NONE")
    parser.add_argument(
        "--output_node", type=str, default="softmax", help="e.g., softmax")
    parser.add_argument(
        "--output_data_types",
        type=str,
        default="float32",
        help="e.g., float32|int32")
    parser.add_argument(
        "--output_data_formats",
        type=str,
        default="NHWC",
        help="e.g., NHWC,NONE")
    parser.add_argument(
        "--check_node", type=str, default="softmax", help="e.g., softmax")
    parser.add_argument(
        "--template_dir", type=str, default="", help="template path")
    parser.add_argument(
        "--obfuscate",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="obfuscate model names")
    parser.add_argument(
        "--model_tag",
        type=str,
        default="",
        help="model tag for generated function and namespace")
    parser.add_argument(
        "--winograd",
        type=int,
        default=0,
        help="Which version of winograd convolution to use. [2 | 4]")
    parser.add_argument(
        "--pattern_style_count",
        type=int,
        default=12,
        help="How many style we have")
    parser.add_argument(
        "--conv_unroll",
        type=int,
        default=1,
        help="CSR conv unroll count")
    parser.add_argument(
        "--dsp_mode", type=int, default=0, help="dsp run mode, defalut=0")
    parser.add_argument(
        "--input_shape", type=str, default="", help="input shape.")
    parser.add_argument(
        "--input_range", type=str, default="", help="input range.")
    parser.add_argument(
        "--output_shape", type=str, default="", help="output shape.")
    parser.add_argument(
        "--check_shape", type=str, default="", help="check shape.")
    parser.add_argument(
        "--platform",
        type=str,
        default="tensorflow",
        help="tensorflow/caffe/onnx")
    parser.add_argument(
        "--pattern_weight",
        type=int,
        default=0,
        help="is pattern weight. 0(Loose)|1(Tight)")
    parser.add_argument(
        "--pattern_config_path",
        type=str,
        default="",
        help="Pattern config path")
    parser.add_argument(
        "--data_type",
        type=str,
        default="float16_float32",
        help="float16_float32/float32_float32")
    parser.add_argument(
        "--model_type",
        type=str,
        default="default",
        help="default/bert")
    parser.add_argument(
        "--graph_optimize_options",
        type=str,
        default="",
        help="graph optimize options")
    parser.add_argument(
        "--quantize",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="quantize model")
    parser.add_argument(
        "--load_from_im",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="load from last results")
    parser.add_argument(
        "--quantize_range_file",
        type=str,
        default="",
        help="file path of quantize range for each tensor")
    parser.add_argument(
        "--pruning_type",
        type=str,
        default="",
        help="PruningType, DENSE, PATTERN, COLUMN")
    parser.add_argument(
        "--executing_devices",
        type=str,
        default="",
        help="Specify operations executing on special devices")
    parser.add_argument(
        "--change_concat_ranges",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="change ranges to use memcpy for quantized concat")
    parser.add_argument(
        "--cl_mem_type",
        type=str,
        default="image",
        help="which memory type to use.[image|buffer]")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
