import argparse
import os
import os.path
import numpy as np
import tempfile
import six
import onnxruntime
import onnxruntime.backend as ort_backend

from lothar.commons import common

"""
python lothar/validate.py --platform=onnx \
    --model_file=/Users/newway/Documents/Research/MobiCom19/DeepVan-All/deep-sparse/models/yolo/yolov3_retrained_0.460_13_striped.onnx \
    --input_file=/Users/newway/Desktop/deepvan_run/model_input \
    --deepvan_out_file=/Users/newway/Desktop/deepvan_run/model_out \
    --input_node=input.1 \
    --output_node=657,727,797 \
    --input_shape=1,416,416,3 \
    --output_shape=507,85:2028,85:8112,85 \
    --output_data_format=NHWC,NHWC,NHWC \
    --backend=pytorch \
    --validation_threshold 0.995
"""
# Validation Flow:
# 1. Generate input data
# 2. Use deepvan_run to run model on phone.
# 3. adb pull the result.
# 4. Compare output data of deepvan and tf
#    python validate.py --model_file tf_model_opt.pb \
#        --input_file input_file \
#        --deepvan_out_file output_file \
#        --input_node input_node \
#        --output_node output_node \
#        --input_shape 1,64,64,3 \
#        --output_shape 1,64,64,2
#        --validation_threshold 0.995

VALIDATION_MODULE = 'VALIDATION'


def load_data(file, data_type='float32'):
    if os.path.isfile(file):
        if data_type == 'float32':
            return np.fromfile(file=file, dtype=np.float32)
        elif data_type == 'int32':
            return np.fromfile(file=file, dtype=np.int32)
        elif data_type == 'int64':
            return np.fromfile(file=file, dtype=np.int32).astype('int64')
    return np.empty([0])


def calculate_sqnr(expected, actual):
    noise = expected - actual

    def power_sum(xs):
        # return sum([x * x for x in xs])
        return np.sum(xs**2)

    signal_power_sum = power_sum(expected)
    noise_power_sum = power_sum(noise)
    return signal_power_sum / (noise_power_sum + 1e-15)


def calculate_similarity(u, v, data_type=np.float64):
    if u.dtype is not data_type:
        u = u.astype(data_type)
    if v.dtype is not data_type:
        v = v.astype(data_type)
    unorm = np.linalg.norm(u)
    vnorm = np.linalg.norm(v)
    if unorm < 1e-3 and vnorm < 1e-3: # magic number
        return 1.0
    if unorm == 0:
        unorm += 1e-15
    if vnorm == 0:
        vnorm += 1e-15
    return np.dot(u, v) / (unorm * vnorm)


def calculate_pixel_accuracy(out_value, deepvan_out_value):
    if len(out_value.shape) < 2:
        # is_equal = lambda x, y: abs(x - y) < abs(x + y) * 1e-1
        is_equal = lambda x, y: abs(x - y) < 1e-4
        correct_count = sum(is_equal(x, y) for x, y in zip(out_value, deepvan_out_value))
        return 1.0 * correct_count / out_value.shape[0]
    out_value = out_value.reshape((-1, out_value.shape[-1]))
    batches = out_value.shape[0]
    classes = out_value.shape[1]
    deepvan_out_value = deepvan_out_value.reshape((batches, classes))
    correct_count = 0
    for i in range(batches):
        if np.argmax(out_value[i]) == np.argmax(deepvan_out_value[i]):
            correct_count += 1
    return 1.0 * correct_count / batches

def compare_output(platform, device_type, output_name, deepvan_out_value,
                   out_value, validation_threshold, log_file):
    max_abs_diff = 0
    max_abs_x = 0
    max_abs_y=0
    if deepvan_out_value.size != 0:
        pixel_accuracy = calculate_pixel_accuracy(out_value, deepvan_out_value)
        print(f'Pixel Accuracy = {pixel_accuracy}')
        if pixel_accuracy < 1 or len(out_value.shape) <= 2:
            # check more when pixel_accuracy is less than a threshold
            temp_out_value = list(out_value.flat)
            temp_deepvan_out_value = list(deepvan_out_value.flat)
            is_equal = lambda x, y: abs(x - y) < min(
                abs(x) / 100, abs(y) / 100) or abs(x - y) < 1e-4
            not_same = sum(not is_equal(x, y) for x, y in zip(temp_out_value, temp_deepvan_out_value))
            for x, y in zip(temp_out_value, temp_deepvan_out_value):
                if not is_equal(x, y) and abs(y-x) > max_abs_diff:
                    max_abs_diff = abs(y-x)
                    max_abs_x = x
                    max_abs_y = y
            if max_abs_diff != 0:
                print(f'deepvan_out = {max_abs_y} != out = {max_abs_x},abs({max_abs_y}-{max_abs_x} = {max_abs_diff})')

            total_count = len(temp_deepvan_out_value)
            common.DLogger.summary(common.StringFormatter.block(
                "pixel by pixel mismatch/total: %s/%s, mismatch rate: %s %%" % 
                    (not_same, total_count, round(not_same * 100 / total_count, 4))))

        out_value = out_value.reshape(-1)
        deepvan_out_value = deepvan_out_value.reshape(-1)
        assert len(out_value) == len(deepvan_out_value)
        sqnr = calculate_sqnr(out_value, deepvan_out_value)
        similarity = calculate_similarity(out_value, deepvan_out_value)
        common.DLogger.summary(
            output_name + ' DEEPVAN VS ' + platform.upper()
            + ' similarity: ' + str(similarity) + ' , sqnr: ' + str(sqnr)
            + ' , pixel_accuracy: ' + str(pixel_accuracy))
        if log_file:
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write('output_name,similarity,sqnr,pixel_accuracy\n')
            summary = '{output_name},{similarity},{sqnr},{pixel_accuracy}\n'\
                .format(output_name=output_name,
                        similarity=similarity,
                        sqnr=sqnr,
                        pixel_accuracy=pixel_accuracy)
            with open(log_file, "a") as f:
                f.write(summary)
        elif similarity > validation_threshold:
            common.DLogger.summary(
                common.StringFormatter.block("Similarity Test Passed"))
        else:
            # common.DLogger.error(
            #     "", common.StringFormatter.block("Similarity Test Failed"))
            common.DLogger.warning(
                common.StringFormatter.block("Similarity Test Failed"))
    else:
        common.DLogger.error(
            "", common.StringFormatter.block(
                "Similarity Test failed because of empty output"))


def normalize_tf_tensor_name(name):
    if name.find(':') == -1:
        return name + ':0'
    else:
        return name


def validate_with_file(platform, device_type,
                       output_names, output_shapes,
                       deepvan_out_file, validation_outputs_data,
                       validation_threshold, log_file):
    for i in range(len(output_names)):
        if validation_outputs_data[i].startswith("http://") or \
                validation_outputs_data[i].startswith("https://"):
            validation_file_name = common.formatted_file_name(
                deepvan_out_file, output_names[i] + '_validation')
            six.moves.urllib.request.urlretrieve(validation_outputs_data[i],
                                                 validation_file_name)
        else:
            validation_file_name = validation_outputs_data[i]
        value = load_data(validation_file_name)
        out_shape = output_shapes[i]
        # if len(out_shape) == 4:
        #     out_shape[1], out_shape[2], out_shape[3] = \
        #         out_shape[3], out_shape[1], out_shape[2]
            # value = value.reshape(out_shape).transpose((0, 2, 3, 1))
        output_file_name = common.formatted_file_name(
            deepvan_out_file, output_names[i])
        deepvan_out_value = load_data(output_file_name)
        compare_output(platform, device_type, output_names[i], deepvan_out_value,
                       value, validation_threshold, log_file)


def validate(platform, model_file, weight_file, input_file, deepvan_out_file,
             device_type, input_shape, output_shape, input_data_format_str,
             output_data_format_str, input_node, output_node,
             validation_threshold, input_data_type, backend,
             validation_outputs_data, log_file):
    input_names = [name for name in input_node.split(',')]
    input_shape_strs = [shape for shape in input_shape.split(':')]
    input_shapes = [[int(x) for x in common.split_shape(shape)]
                    for shape in input_shape_strs]
    output_shape_strs = [shape for shape in output_shape.split(':')]
    output_shapes = [[int(x) for x in common.split_shape(shape)]
                     for shape in output_shape_strs]
    input_data_formats = [df for df in input_data_format_str.split(',')]
    output_data_formats = [df for df in output_data_format_str.split(',')]
    if input_data_type:
        input_data_types = [data_type for data_type in input_data_type.split(',')]
    else:
        input_data_types = ['float32'] * len(input_names)
    output_names = [name for name in output_node.split(',')]
    assert len(input_names) == len(input_shapes)
    if not isinstance(validation_outputs_data, list):
        if os.path.isfile(validation_outputs_data):
            validation_outputs = [validation_outputs_data]
        else:
            validation_outputs = []
    else:
        validation_outputs = validation_outputs_data
    if validation_outputs:
        validate_with_file(platform, device_type, output_names, output_shapes,
                           deepvan_out_file, validation_outputs,
                           validation_threshold, log_file)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", type=str, default="", help="TensorFlow or Caffe.")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="TensorFlow or Caffe \'GraphDef\' file to load.")
    parser.add_argument(
        "--weight_file",
        type=str,
        default="",
        help="caffe model file to load.")
    parser.add_argument(
        "--input_file", type=str, default="", help="input file.")
    parser.add_argument(
        "--deepvan_out_file",
        type=str,
        default="",
        help="deepvan output file to load.")
    parser.add_argument(
        "--device_type", type=str, default="", help="deepvan runtime device.")
    parser.add_argument(
        "--input_shape", type=str, default="1,64,64,3", help="input shape.")
    parser.add_argument(
        "--input_data_format", type=str, default="NHWC",
        help="input data format.")
    parser.add_argument(
        "--output_shape", type=str, default="1,64,64,2", help="output shape.")
    parser.add_argument(
        "--output_data_format", type=str, default="NHWC",
        help="output data format.")
    parser.add_argument(
        "--input_node", type=str, default="input_node", help="input node")
    parser.add_argument(
        "--input_data_type",
        type=str,
        default="",
        help="input data type")
    parser.add_argument(
        "--output_node", type=str, default="output_node", help="output node")
    parser.add_argument(
        "--validation_threshold", type=float, default=0.995,
        help="validation similarity threshold")
    parser.add_argument(
        "--backend",
        type=str,
        default="tensorflow",
        help="onnx backend framwork")
    parser.add_argument(
        "--validation_outputs_data", type=str,
        default="", help="validation outputs data file path.")
    parser.add_argument(
        "--log_file", type=str, default="", help="log file.")

    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    validate(FLAGS.platform,
             FLAGS.model_file,
             FLAGS.weight_file,
             FLAGS.input_file,
             FLAGS.deepvan_out_file,
             FLAGS.device_type,
             FLAGS.input_shape,
             FLAGS.output_shape,
             FLAGS.input_data_format,
             FLAGS.output_data_format,
             FLAGS.input_node,
             FLAGS.output_node,
             FLAGS.validation_threshold,
             FLAGS.input_data_type,
             FLAGS.backend,
             FLAGS.validation_outputs_data,
             FLAGS.log_file)
