import numpy as np

from deepvan.proto import deepvan_pb2
from lothar.net_converter import base_converter as cvt
from lothar.net_converter.base_converter import DeepvanConfigKey
from lothar.net_converter.base_converter import DeviceType
from lothar.net_converter.convert_util import CONDITIONS


class TensorInfo:
    def __init__(self, id, tensor):
        self.id = id
        self.data_type = tensor.data_type
        if tensor.data_type == deepvan_pb2.DT_HALF:
            self.data = convert_data_to(tensor.float_data, np.float16)
        elif tensor.data_type == deepvan_pb2.DT_FLOAT:
            self.data = convert_data_to(tensor.float_data, np.float32)
        elif tensor.data_type == deepvan_pb2.DT_INT32:
            self.data = convert_data_to(tensor.int32_data, np.int32)
        elif tensor.data_type == deepvan_pb2.DT_UINT8:
            self.data = bytearray(
                np.array(tensor.int32_data).astype(np.uint8).tolist())
        elif tensor.data_type == deepvan_pb2.DT_INT8:
            self.data = bytearray(
                np.array(tensor.int32_data).astype(np.uint8).tolist())
        else:
            raise Exception('Tensor data type %s not supported' %
                            tensor.data_type)


def convert_data_to(tensor_data, data_type):
    if data_type == np.uint8:
        return bytearray(np.array(tensor_data).astype(data_type).tolist())
    if data_type == np.float16:
        if np.max(tensor_data) > np.finfo(data_type).max or np.min(tensor_data) < np.finfo(data_type).min:
            print(
                "Warning: Clip values ​​before cast to fp16 in case an overflow issue is detected.")
            tensor_data = np.clip(np.array(tensor_data), np.finfo(
                data_type).min, np.finfo(data_type).max)
    return bytearray(np.array(tensor_data).astype(data_type).tobytes())


class ModelPersistent(object):
    def __init__(self, option, net_def, pruning_type):
        self._option = option
        self._net_def = net_def
        self._pruning_type = pruning_type
        self._data_type = option.data_type
        self._pattern_infos = []
        self._tensor_infos = []

    def set_tensor_size(self, tensor):
        if tensor.data_type == deepvan_pb2.DT_FLOAT \
                or tensor.data_type == deepvan_pb2.DT_HALF:
            tensor.data_size = len(tensor.float_data)
        elif tensor.data_type == deepvan_pb2.DT_INT32:
            tensor.data_size = len(tensor.int32_data)
        elif tensor.data_type == deepvan_pb2.DT_UINT8:
            tensor.data_size = len(tensor.int32_data)
        elif tensor.data_type == deepvan_pb2.DT_INT8:
            tensor.data_size = len(tensor.int32_data)
        return tensor.data_size

    def update_tensor_infos(self):
        """
        Set tensor offsets and other information
        """
        cpu_tensors = []
        for op in self._net_def.op:
            for arg in op.arg:
                if arg.name == DeepvanConfigKey.deepvan_executing_on_str:
                    if len(arg.ints) == 1 and DeviceType.CPU.value in arg.ints:
                        cpu_tensors.extend(op.input)
        offset = 0
        counter = 0
        tensor_size = 0
        for tensor in self._net_def.tensors:
            tensor_size += len(tensor.float_data)
            if tensor.data_type == deepvan_pb2.DT_FLOAT and tensor.name not in cpu_tensors:
                tensor.data_type = self._data_type
            if not tensor.float_data and not tensor.int32_data:
                continue
            # Add offset and data_size
            tensor_info = TensorInfo(counter, tensor)
            self._tensor_infos.append(tensor_info)
            # align
            if tensor_info.data_type != deepvan_pb2.DT_UINT8 \
                    and tensor_info.data_type != deepvan_pb2.DT_INT8 \
                    and offset % 4 != 0:
                padding = 4 - offset % 4
                offset += padding

            self.set_tensor_size(tensor)
            tensor.offset = offset
            offset += len(tensor_info.data)
            counter += 1

    def extract_model_data(self):
        """
        Extract model data
        """
        model_data = []
        offset = 0
        counter = 0
        total_size = 0
        for tensor in self._net_def.tensors:
            if not tensor.float_data and not tensor.int32_data:
                continue
            tensor_info = TensorInfo(counter, tensor)
            # align
            CONDITIONS(offset <= tensor.offset,
                       "Current offset should be <= tensor.offset")
            if offset < tensor.offset:
                model_data.extend(bytearray([0] * (tensor.offset - offset)))
                offset = tensor.offset
            model_data.extend(tensor_info.data)
            offset += len(tensor_info.data)
            counter += 1
            total_size += len(tensor_info.data)
        return model_data

    def save_model_data(self, model_tag, output_dir):
        model_data = self.extract_model_data()
        # generate tensor data
        with open(output_dir + model_tag + '.data', "wb") as f:
            f.write(bytearray(model_data))

    def save_model_to_proto(self, model_tag, output_dir):
        for tensor in self._net_def.tensors:
            if tensor.data_type == deepvan_pb2.DT_FLOAT \
                    or tensor.data_type == deepvan_pb2.DT_HALF:
                del tensor.float_data[:]
            elif tensor.data_type == deepvan_pb2.DT_INT32:
                del tensor.int32_data[:]
            elif tensor.data_type == deepvan_pb2.DT_UINT8:
                del tensor.int32_data[:]
            elif tensor.data_type == deepvan_pb2.DT_INT8:
                del tensor.int32_data[:]
            del tensor.row_ptr[:]
            del tensor.col_index[:]
            # del pattern data
            del tensor.pattern_data.number_data[:]
            del tensor.pattern_data.order_data[:]
            if tensor.pattern_data.pst == deepvan_pb2.PT_LOOSE:
                del tensor.pattern_data.style_data[:]
            elif tensor.pattern_data.pst == deepvan_pb2.PT_TIGHT:
                del tensor.pattern_data.index_data[:]
                del tensor.pattern_data.gap_data[:]
            # del csr col information
            del tensor.csr_data.col_ptr[:]
            del tensor.csr_data.row_ptr[:]
            # del slice information
            # del tensor.slice_data.order_data[:]
            # del tensor.slice_data.index_data[:]
            # del tensor.slice_data.offset_data[:]

        proto_file_path = output_dir + model_tag + '.pb'
        with open(proto_file_path, "wb") as f:
            f.write(self._net_def.SerializeToString())
        with open(proto_file_path + '_txt', "w") as f:
            f.write(str(self._net_def))

        return proto_file_path


def check_tensors(model):
    for tensor in [t for t in model.tensors if t.pruning_type == deepvan_pb2.CSR]:
        print("Tensor shape: %s, data_size: %s, data offset: %s, row size: %s, row offset: %s, col size: %s, col offset: %s" % (
            tensor.dims, tensor.data_size, tensor.offset, tensor.csr_data.row_size, tensor.csr_data.row_offset, tensor.csr_data.col_size, tensor.csr_data.col_offset))


def save_model(option, net_def, template_dir,
               obfuscate, model_tag, output_dir,
               winograd_conv, pruning_type=0):
    for tensor in [t for t in net_def.tensors if t.pruning_type == deepvan_pb2.CSR]:
        print("Before: Tensor shape: %s, data: %s, row: %s, col: %s" %
              (tensor.dims, tensor.float_data[:6], tensor.csr_data.row_ptr[:10], tensor.csr_data.col_ptr[:6]))
    persistenter = ModelPersistent(option, net_def, pruning_type)
    output_dir = output_dir + '/'
    # update tensor type
    cvt.timer_wrapper(None, persistenter.update_tensor_infos)

    cvt.timer_wrapper(None, persistenter.save_model_data,
                      model_tag, output_dir)
    cvt.timer_wrapper(None, persistenter.save_model_to_proto,
                      model_tag, output_dir)
    check_tensors(net_def)
