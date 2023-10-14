#ifndef DEEPVAN_EXPORT_XGEN_H_
#define DEEPVAN_EXPORT_XGEN_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XGEN_EXPORT __attribute__((visibility("default")))

typedef struct XGenHandle XGenHandle;
typedef struct XGenTensor XGenTensor;

typedef enum XGenStatus {
  XGenOk = 0,
  XGenError = 1,
  XGenLicenseExpired = 2,
} XGenStatus;

typedef enum XGenType {
  XGenNone = 0,
  XGenFloat32 = 1,
  XGenFloat16 = 2,
  XGenInt32 = 3,
  XGenInt8 = 4,
  XGenUInt8 = 5,

  XGenNumTypes = 6,
} XGenType;

typedef enum XGenPowerPolicy {
  /* No power policy. Use whatever system default configuration is. */
  XGenPowerNone = 0,
  /* Default power policy is used with XGenInitWithData */
  XGenPowerDefault = 1,
  /* Performance policy uses less power than the default but still performs
     well. */
  XGenPowerPerformance = 2,
  /* Power save policy uses even less power than performance policy. */
  XGenPowerSave = 3,
} XGenPowerPolicy;

/**
 * Initialize XGen in DeepOpt mode.
 * Arguments, such as model_file, data_file and their lengths, can
 * be found in generated *.pb and *.data respectively.
 */
XGEN_EXPORT XGenHandle *XGenInitWithFiles(const char *model_file,
                                          const char *data_file,
                                          XGenPowerPolicy policy = XGenPowerDefault);

/**
 * Initialize XGen in DeepOpt mode.
 * Arguments, such as model_data, extra_data and their lengths, can
 * be found in generated xgen_pb.h and xgen_data.h respectively.
 */
XGEN_EXPORT XGenHandle *XGenInitWithData(const void *model_data,
                                         size_t model_size_in_bytes,
                                         const void *extra_data,
                                         size_t data_size_in_bytes);

/**
 * Initialize XGen in DeepOpt mode with a power policy.
 * Arguments, such as model_data, extra_data and their lengths, can
 * be found in generated xgen_pb.h and xgen_data.h respectively.
 */
XGEN_EXPORT XGenHandle *XGenInitWithPower(const void *model_data,
                                          size_t model_size_in_bytes,
                                          const void *extra_data,
                                          size_t data_size_in_bytes,
                                          XGenPowerPolicy policy);

/**
 * Initialize XGen in Fallback mode.
 * Arguments can be found in generated xgen_fallback.h. The lifetime
 * of the `model_data` must be at least as long as the lifetime of
 * returned `XGenHandle`
 */
XGEN_EXPORT XGenHandle *XGenInit(const void *model_data, size_t model_size);

/**
 * Initialize XGen in Fallback mode and run models only on CPU.
 * Arguments can be found in generated xgen_fallback.h. The lifetime
 * of the `model_data` must be at least as long as the lifetime of
 * returned `XGenHandle`
 */
XGEN_EXPORT XGenHandle *XGenInitWithCPUOnly(const void *model_data,
                                            size_t model_size);

/**
 * Initialize XGen with model files.
 * In Fallback mode, data_path should be nullptr. The lifetime
 * of the `model_data` must be at least as long as the lifetime of
 * returned `XGenHandle`
 */
XGEN_EXPORT XGenHandle *XGenInitWithFallbackFiles(const char *model_path);
/**
 * Run the model.
 */
XGEN_EXPORT XGenStatus XGenRun(XGenHandle *handle);

/**
 * Shutdown XGen and release allocated resources.
 */
XGEN_EXPORT void XGenShutdown(XGenHandle *handle);

/**
 * Return number of input tensors in the model
 */
XGEN_EXPORT size_t XGenGetNumInputTensors(const XGenHandle *handle);

/**
 * Return number of output tensors in the model
 */
XGEN_EXPORT size_t XGenGetNumOutputTensors(const XGenHandle *handle);

/**
 * Return the input tensor indexed by tensor_index in the model
 */
XGEN_EXPORT XGenTensor *XGenGetInputTensor(XGenHandle *handle,
                                           size_t tensor_index);

/**
 * Return the output tensor indexed by tensor_index in the model
 */
XGEN_EXPORT XGenTensor *XGenGetOutputTensor(XGenHandle *handle,
                                            size_t tensor_index);

/**
 * Initialize the input tensor with provided data.
 */
XGEN_EXPORT void XGenCopyBufferToTensor(XGenTensor *input_tensor,
                                        const void *input_data,
                                        size_t input_size_in_bytes);
/**
 * Retrieve output data from the output tensor.
 * @param output_data pre-allocated buffer.
 * @param output_size_in_bytes number of bytes to copy into output_data.
 */
XGEN_EXPORT void XGenCopyTensorToBuffer(const XGenTensor *output_tensor,
                                        void *output_data,
                                        size_t output_size_in_bytes);

/**
 * Return the type of the tensor.
 */
XGEN_EXPORT XGenType XGenGetTensorType(const XGenTensor *tensor);

/**
 * Return number of dimensions.
 */
XGEN_EXPORT int32_t XGenGetTensorNumDims(const XGenTensor *tensor);

/**
 * Return the size of the dimension indexed by dim_index.
 */
XGEN_EXPORT int32_t XGenGetTensorDim(const XGenTensor *tensor,
                                     int32_t dim_index);

/**
 * Return the size of the tensor in bytes.
 */
XGEN_EXPORT size_t XGenGetTensorSizeInBytes(const XGenTensor *tensor);

/**
 * Return the pointer to the underlying data.
 */
XGEN_EXPORT void *XGenGetTensorData(const XGenTensor *tensor);

/**
 * Return the name of the tensor.
 */
XGEN_EXPORT const char *XGenGetTensorName(const XGenTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
