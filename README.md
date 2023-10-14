# JPEG-AI Verification Model 3.3

## Introduction
JPEG-AI is a learning-based image coding standard, and it includes verification models as part of the standard. This repository aims to offer the verification models in ONNX format, along with our optimized backend support known as JAIOPT. These models consist of five pairs of Deep Neural Network (DNN) models used within the JPEG-AI framework, including single_encode, hyper_encoder, hyper_decoder, hyper_scale_decoder, and single_decoder. Each of these pairs includes two models, one for luminance (_y) and one for chrominance (_uv) encoding or decoding. The first four pairs of models are employed in the encoding phase, while the last three pairs are used in the decoding phase.

It is essential to note that the ONNX models, as well as our inference framework found in this repository, are primarily intended for performance evaluation rather than end-to-end inference. Additionally, we provide performance comparisons between our framework and others. In summary, our inference framework demonstrates significant speed improvements, approximately 5.5-6.5 times faster than NNAPI and about 3 times faster than QNN (Qualcomm Neural Network).

## Structure of this Repo
This repository comprises two directories: `JAIOPT`, which has our inference framework including compiler and inference engine, and `Verification_Models`, which contains the models discussed in the `Introduction`.

## Performance Benchmarking
Table 1 provides performance comparisons of decoder models between our inference framework and others on Snapdragon SoC (especially GPU) with Android OS. The focus is on execution latency. The testing configurations are as follows:

* Input image size: 1024x1024                                                          
* Benchmark platform: Snapdragon 8 Gen 2, GPU                              
* Inference framework: JAIOPT (i.e., our framework), NNAPI (with TFLite and ONNX-Runtime, e.g., ORT as front ends) and QNN
* Data type: FP16    


|     Model                     |     NNAPI   w/ONNX (ms)    |     NNAPI   w/ tflite   (ms)    |     QNN-GPU   (ms)    |     JAIOPT   (ms)    |     Speedup      over ORT    |     Speedup     over Tflite    |     Speedup     over   QNN-GPU    |
|-------------------------------|----------------------------|---------------------------------|-----------------------|--------------------|------------------------------|--------------------------------|-----------------------------------|
|     decoder_uv                |            41.95           |               33.54             |          26.32        |         6.42       |             6.53x            |              5.22x             |                4.1x               |
|     decoder_y                 |            171.67          |              142.57             |          38.97        |        19.09       |             8.99x            |              7.47x             |                2.04x              |
|     hyper_decoder_uv          |             1.89           |               1.48              |          4.29         |         2.17       |             0.87x            |              0.68x             |                1.98x              |
|     hyper_decoder_y           |            19.15           |               22.61             |          22.07        |         5.14       |             3.73x            |               4.4x             |                4.3x               |
|     hyper_scale_decoder_uv    |             1.93           |               1.46              |          3.93         |         2.15       |              0.9x            |              0.68x             |                1.83x              |
|     hyper_scale_decoder_y     |             18.9           |               22.27             |          21.41        |         5.08       |             3.72x            |              4.39x             |                4.22x              |
|     Total                     |            255.49          |              223.94             |           117         |        40.04       |             6.38x            |              5.59x             |                2.92x              |

**Table 1. DECODER Performance Comparisons**


## Environment Setup
Please adhere to the detailed instructions in the following sub-section for setting up the environment prior to conducting performance measurements.

### 1. Python path setup
  Update the python path to the path of your python3.7 (Python version later than 3.7 should work as well) in file repo/opencl-kernel/opencl_kernel_configure.bzl
  by changing the line `python_bin_path = "/usr/local/bin/python3.7"`
  You can see your Python path by run 
  `whereis python`

> In the following, whenever you see any importing error, you may need to install the missing package by running `pip install package_name`

### 2. Install Boost lib
On Linux, run:
```bash
sudo apt-get install libboost-all-dev
```

### 3. Android SDK/NDK setup
- Install Bazel (5.0.0) from [Bazel Documentation](https://docs.bazel.build/versions/master/install.html)
- Download and install Android SDK. It can be downloaded either from [Android Studio](https://developer.android.com/studio) or from the Android SDK command line tool (https://developer.android.com/studio#command-tools).
- Download [Android NDK](https://developer.android.com/ndk/downloads) version `r16b` or `r17c` (later versions may be supported but have not been tested)
- Export the directory of Android SDK and Android NDK to the environment path
  For instance, on Mac cshell, add the following into .cshrc (change the paths to yours), and on Linux, you can add corresponding paths to .bashrc.
  `setenv ANDROID_SDK_HOME ~/Library/Android/sdk`
  `setenv ANDROID_NDK_HOME ~/Programs/android-ndk-r17c`
  `setenv PATH ~/Library/Android/sdk/tools:~/Library/Android/sdk/platform-tools:$PATH`
  
### 4. Install this Compiler Toolkit
If you have already cloned this repository, the compilation toolkit should be in your possession. If not, please download it from this repository.

### 5. Check your setup
Enter the `JAIOPT` directory. To check whether your setup is in good shape, run the following in the root directory of the toolkit.

`bazel build --config android --config optimization //deepvan/executor:libexecutor_shared.so --config symbol_hidden --define neon=true --define openmp=true --define opencl=true --cpu=arm64-v8a`

## How to measure model performance
Let us use the `decoder_uv` model as an example for performance measurement. Navigate to the `Verification_Models` directory to locate the `decoder_uv` model, and take note that there is a configuration file named `decoder_uv.yml` associated with this model.

We need to perform the following two steps:                                                                     

1. Conversion

Execute command `python3.7 TOOL_PATH/lothar/controller.py convert --config=path/to/your/config.yml --model_path=ONNX_PATH` to convert that ONNX to our internal computational graph.

In the provided command line, replace `TOOL_PATH` with the root path of this toolkit, and `ONNX_PATH` with the path to your testing ONNX model. For the `config` option, use the predefined `decoder_uv.yml` file, which contains pre-defined parameters like model input/output shape, data type, and runtime. The template config.yml can be referred to [here](https://github.com/hustc12/jpeg-ai-release/blob/main/config.yml).

2. Run
  
To run the model, connect your Android phone to this computer, enable `Developer` mode, and turn on `USB Debugging` in the `Developer Options` within your phone's Settings. Afterward, run the following command in the root directory of this tool on your computer:

  `python3.7 TOOL_PATH/lothar/controller.py run --config=path/to/your/config.yml --model_path=ONNX_PATH`
 where TOOL_PATH and ONNX_PATH should be replaced as above.

 The `decoder_uv` model will then be executed on the smartphone on some random inputs created by the script.

## License

  Source code uses [Apache License 2.0](https://github.com/SmartHarmony/JPEG-AI-OPT/blob/main/LICENSE)
  
  Other materials uses [Creative Commons Attribution (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/)

















