# JPEG-AI Verification Model 3.3

## Introduction
JPEG-AI is a learning-based image coding standard, and it includes verification models as part of the standard. This repository aims to offer the verification models in ONNX format, along with our optimized backend support known as JAIOPT. These models consist of five pairs of Deep Neural Network (DNN) models used within the JPEG-AI framework, including single_encode, hyper_encoder, hyper_decoder, hyper_scale_decoder, and single_decoder. Each of these pairs includes two models, one for luminance (_y) and one for chrominance (_uv) encoding or decoding. The first four pairs of models are employed in the encoding phase, while the last three pairs are used in the decoding phase.

It is essential to note that the ONNX models, as well as our inference framework found in this repository, are primarily intended for performance evaluation rather than end-to-end inference. Additionally, we provide performance comparisons between our framework and others. In summary, our inference framework demonstrates significant speed improvements, approximately 5.5-6.5 times faster than NNAPI and about 3 times faster than QNN (Qualcomm Neural Network).

The backend in this repo was tested on Qualcomm Sanpdragon 8 only. It may not support other hardware well. 

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

## Result Verification
We conducted a comparison between the computed results of the exported (ONNX) models and the original PyTorch version on a pixel level, setting tolerance thresholds ranging from 0.5% to 1%. The results indicate that the exported ONNX models consistently yield accurate results when compared to the PyTorch version.

Diagram 1 illustrates the verification procedure's underlying mechanism.

![Diagram 1.](/images/Correctness_Verification.drawio.svg "This is a sample image.")

**Diagram 1. Result Verification Procedure**

## Supported Operator Types
Table 2 displays the operator types that are supported and have been utilized in the verification models.

| Operator Type   |
|-----------------|
| Activation      |
| BufferTransform |
| Conv2D          |
| Deconv2D        |
| Eltwise         |

**Table 2. Supported Operator Types**

## Environment Setup
Please adhere to the detailed instructions in the following sub-section for setting up the environment prior to conducting performance measurements.

### 1. Install our Compiler Toolkit
Firstly, you need to clone the compiler toolkit from our GitHub repository. 
If you have already cloned this repository, the compilation toolkit should be in your possession. If not, please download it from this repository by the command:
```bash
git clone https://github.com/SmartHarmony/JPEG-AI-OPT
```

### 2. Set up Python path and libraries
First, configure the Python path to align with your specific environment.
> *Note*: This instruction uses `Python 3.7` as an example. However, Python versions later than 3.7 should also be compatible.  Make sure to use the path corresponding to the Python version you are working with.

Enter `JAIOPT` and navigate to the `repo/opencl-kernel` directory. Open the file named `opencl_kernel_configure.bzl` and find line 40 in the file. It should look like this: 
```
python_bin_path = "/usr/local/bin/python3.7"
```

Replace `/usr/local/bin/python3.7` with the path to your `Python 3.7` installation.

To find your Python path, run the following command in your terminal:
  ```bash
whereis python
```

  
After configuring your custom Python path, the next step is to set up the required libraries. Return to the `JAIOPT` directory. There, you can install the necessary Python libraries by executing the following command:
```bash
pip install -r requirements.txt
``` 

> *Note:* Whenever you see any importing error, you may need to install the missing package by running
> ```bash
> pip install missing_package_name
> ```

### 3. Install required build tools and packages
On **Linux**, run:
```bash
sudo apt-get install cmake gcc g++ libboost-all-dev libncurses5
```
On **MacOS**, run:
```bash
brew install cmake gcc boost ncurses
```
For `MacOS` users, if Homebrew is not installed on your system, you can install it by the instructions on the [Homebrew website](https://brew.sh/). 

### 4. Set up Android SDK/NDK
- Install Bazel (5.0.0) from [Bazel Documentation](https://docs.bazel.build/versions/master/install.html).
- Download and install Android SDK. It can be downloaded either from [Android Studio](https://developer.android.com/studio) or from the Android SDK [command line tool](https://developer.android.com/studio#command-tools).
- Download [Android NDK](https://developer.android.com/ndk/downloads) version `r16b` or `r17c` (later versions may be supported but have not been tested).
- Export the directory of Android SDK and Android NDK to the environment path.

  If you are in `zsh` or `bash` environments, add the following lines to your `.bashrc` file:
  ```bash
  export ANDROID_SDK_HOME=~/path/to/Android/sdk
  export ANDROID_NDK_HOME=~/path/to/android-ndk-r17c
  export PATH=~/path/to/Android/sdk/tools:~/path/to/Android/sdk/platform-tools:$PATH
  ```

  If you are in `cshell` enviornments, add the following lines to your `.cshrc` file:
  ```
  setenv ANDROID_SDK_HOME ~/path/to/Android/sdk
  setenv ANDROID_NDK_HOME ~/path/to/android-ndk-r17c
  setenv PATH ~/path/to/Android/sdk/tools:~/path/to/Android/sdk/platform-tools:$PATH`
  ```
  

### 5. Check your setup
To verify that your setup is correctly configured, navigate to the `JAIOPT` folder and execute the following commands:

```
bazel build --config android --config optimization //deepvan/executor:libexecutor_shared.so --config symbol_hidden --define neon=true --define openmp=true --define opencl=true --cpu=arm64-v8a
```

>*Note:* In this step, if you encounter the build error:
>```
>  error: invalid value 'c++17' in '-std=c++17'
>```
> please proceed with the modifications outlined below:
> - Navigate to the `JAIOPT` directory and open the `.bazelrc` file.
> - Locate `lines 8` and `9`, which should appear as follows: 
> ```
>  build --cxxopt=-std=c++17
>  build --host_cxxopt=-std=c++17
> ```
> - Update these lines to use the `C++1z` standard instead of `C++17`:
>   
>```
>  build --cxxopt=-std=c++1z
>  build --host_cxxopt=-std=c++1z
> ```


## How to measure model performance
In this part, we use `decoder_uv` model as an example for performance measurement.

### 1. Build JAIOPT framework

Before measuring performance, build the `JAIOPT` framework. Navigate to the `JAIOPT` directory in your terminal and run the `build.sh` script to initiate the build process. This can be done by executing the command:
```bash
./build.sh
```
### 2. Configuration for model

After compiling `JAIOPT` successfully, proceed with the configuration steps for the `decoder_uv` model:

Navigate to the `Verification_Models/decoder` directory and find the `decoder_uv.yml` configuration file, which contains pre-defined parameters like model input/output shape, data type, and runtime. Update the `model_file_path` in this file to reflect the correct path by modifying `line 6` to:
```
 model_file_path: path/to/model/decoder_uv.onnx
```
>*Note:* This example is specific to the `decoder_uv` model. For a general `.yml` template, please refer to the [config.yml](https://github.com/SmartHarmony/JPEG-AI-OPT/blob/main/config.yml).

### 3. Measure model performance

We need to perform the following two steps:                                                                     

### 3.1 Conversion

Execute command:
```bash
python3.7 path/to/JAIOPT/lothar/controller.py convert --config=path/to/decoder_uv.yml --model_path=path/to/decoder_uv.onnx
```
to convert that ONNX to our internal computational graph.

>*Note:* In the given command, the `decoder_uv` model is used as a representative example. For testing with a different `ONNX` model, replace `decoder_uv` with the name of your target model and update the corresponding configuration `.yml` file accordingly. Refer to the contents of the `Verification_Models` directory for examples of model configurations.

The result of conversion should be like:

**Insert figures of conversion result here.**

#### 3.2 Run
  
To run the model, connect your Android phone to this computer, enable `Developer` mode, and turn on `USB Debugging` in the `Developer Options` within your phone's Settings. Afterward, run the following command in the root directory of this tool on your computer:
```bash
python3.7 path/to/JAIOPT/lothar/controller.py run --config=path/to/decoder_uv.yml --model_path=path/to/decoder_uv.onnx
```
>*Note:* In the given command, the `decoder_uv` model is used as a representative example. For testing with a different `ONNX` model, replace `decoder_uv` with the name of your target model and update the corresponding configuration `.yml` file accordingly. Refer to the contents of the `Verification_Models` directory for examples of model configurations.

The `decoder_uv` model will then be executed on the smartphone on some random inputs created by the script. The running result should be like:

**Insert figures of running result here.**

## License
  Source code uses [Apache License 2.0](https://github.com/SmartHarmony/JPEG-AI-OPT/blob/main/LICENSE)

  Other materials uses [Creative Commons Attribution (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/)
