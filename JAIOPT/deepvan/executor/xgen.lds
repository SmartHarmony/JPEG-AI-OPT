XGEN_VER_1 {
  global:
    XGenInit*;
    XGenRun;
    XGenShutdown;
    XGenGetNumInputTensors;
    XGenGetNumOutputTensors;
    XGenGetInputTensor;
    XGenGetOutputTensor;
    XGenCopyBufferToTensor;
    XGenCopyTensorToBuffer;
    XGenGetTensorType;
    XGenGetTensorNumDims;
    XGenGetTensorDim;
    XGenGetTensorSizeInBytes;
    XGenGetTensorData;
    XGenGetTensorName;
  local: *;
};
