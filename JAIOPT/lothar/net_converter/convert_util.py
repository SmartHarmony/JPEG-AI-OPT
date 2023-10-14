import enum


def CONDITIONS(condition, msg):
    if not condition:
        raise Exception(msg)


class OpenCLBufferType(enum.Enum):
    CONV2D_FILTER = 0
    IN_OUT_CHANNEL = 1
    ARGUMENT = 2
    IN_OUT_HEIGHT = 3
    IN_OUT_WIDTH = 4
    WINOGRAD_FILTER = 5
    DW_CONV2D_FILTER = 6
    WEIGHT_HEIGHT = 7
    WEIGHT_WIDTH = 8
