import os
import numpy as np
import math
import common
import argparse


def load_data(npy_file, datatype):
    data = np.load(npy_file)
    if datatype == 'float32':
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float16)
    return data


def load_data_from_file(file, data_type='float32'):
    if os.path.isfile(file):
        if data_type == 'float32':
            return np.fromfile(file=file, dtype=np.float32)
        elif data_type == 'float16':
            return np.fromfile(file=file, dtype=np.float16)
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
    if unorm < 1e-3 and vnorm < 1e-3:  # magic number
        return 1.0
    if unorm == 0:
        unorm += 1e-15
    if vnorm == 0:
        vnorm += 1e-15
    return np.dot(u, v) / (unorm * vnorm)


def calculate_pixel_accuracy(out_value, deepvan_out_value):
    if len(out_value.shape) < 2:
        def is_equal(x, y): return abs(x - y) < abs(x + y) * 1e-1
        correct_count = sum(is_equal(x, y)
                            for x, y in zip(out_value, deepvan_out_value))
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


def compare_output(output_name, deepvan_out_value,
                   out_value, validation_threshold, log_file):
    if deepvan_out_value.size != 0:
        pixel_accuracy = calculate_pixel_accuracy(out_value, deepvan_out_value)
        if pixel_accuracy < 1 or len(out_value.shape) <= 2:
            # check more when pixel_accuracy is less than a threshold
            temp_out_value = list(out_value.flat)
            temp_deepvan_out_value = list(deepvan_out_value.flat)
            def is_equal(x, y): return abs(x - y) < min(
                abs(x) / 100, abs(y) / 100) or abs(x - y) < 5e-2
            not_same = sum(not is_equal(x, y)
                           for x, y in zip(temp_out_value, temp_deepvan_out_value))
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
            output_name + ' DEEPVAN VS ' + 'PyTorch'
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


def process_input_data(input_npy_data, processed_data_file):
    data = np.load(input_npy_data, allow_pickle=True)
    with open(processed_data_file, 'wb') as fr:
        data.astype(np.float16).tofile(fr)


def convert32_16(data):
    float16_array = data.astype(np.float16)
    return float16_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_results', type=str, default = '', required=True, help = 'Results from the PyTorch model')
    parser.add_argument('--jai_results', type=str, default = '', required=True, help='Results from the JPEG AI model')
    parser.add_argument('--validation_threshold', type=float, default=0.995, help='Validation threshold')
    
    return parser.parse_known_args()



if '__main__' == __name__:
    # np.set_printoptions(threshold=math.inf)
    FLAGS, unparsed = parse_args()
    pytorch_results = FLAGS.pytorch_results
    jai_results = FLAGS.jai_results
    threshold = FLAGS.validation_threshold
    jai_results = load_data_from_file(jai_results)
    pytorch_results = load_data_from_file(pytorch_results)

    # Start to compare the results
    compare_output('testing', jai_results,
                   pytorch_results, float(threshold), False)
