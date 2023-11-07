# JPEG-AI Verification Model 3.3

## Introduction
`JPEG-AI` is a learning-based image coding standard, and it includes verification models as part of the standard. This repository aims to offer the verification models in ONNX format, along with our optimized backend support known as JAIOPT. These models consist of five pairs of **Deep Neural Network (DNN)** models used within the JPEG-AI framework, including `single_encode`, `hyper_encoder`, `hyper_decoder`, `hyper_scale_decoder`, and `single_decoder`. Each of these pairs includes two models, one for **luminance (_y)** and one for **chrominance (_uv)** encoding or decoding. The first four pairs of models are employed in the encoding phase, while the last three pairs are used in the decoding phase.  
It is essential to note that the ONNX models, as well as our inference framework found in this repository, are primarily intended for performance evaluation rather than end-to-end inference. Additionally, we provide performance comparisons between our framework and others.  
`In summary, our inference framework demonstrates significant speed improvements, approximately 5.5-6.5 times faster than NNAPI and about 3 times faster than QNN (Qualcomm Neural Network).`

The backend in this repo was tested on `Qualcomm Sanpdragon 8` device (Example: Samsung Galaxy S21+).  

## Structure of this Repo
This repository comprises two directories:  
1. `JAIOPT`  
Has our inference framework including compiler and inference engine  
2. `Verification_Models`  
Contains the models discussed in the `Introduction`.

```
.
├── config.yml
├── Copyright.md
├── images
├── JAIOPT               // Our inference framework
├── LICENSE-APACHE
├── LICENSE-MIT
├── README.md
└── Verification_Models  // Models
```

## Performance Benchmarking
Table 1 provides performance comparisons of decoder models between our inference framework and others on Snapdragon SoC (especially GPU) with Android OS. The focus is on execution latency. The testing configurations are as follows:

* Input image size:  
`1024x1024`  
* Benchmark platform:  
`Snapdragon 8 Gen 2, GPU`  
* Inference framework:  
`JAIOPT` (i.e., our framework)  
`NNAPI` (with TFLite and ONNX-Runtime, e.g., ORT as front ends)  
`QNN`
* Data type:  
`FP16`


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
We conducted a comparison between the computed results of the exported (ONNX) models and the original PyTorch version on a pixel level, setting tolerance thresholds ranging from `0.5%` to `1%`. The results indicate that the exported ONNX models consistently yield accurate results when compared to the PyTorch version.

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
### 1. Install our Compiler Toolkit
Clone this repository to your workspace.
(Our workspace in this example is `~/Documents/GitHub/demo`)
```bash
mkdir ~/Documents/GitHub/demo
cd ~/Documents/GitHub/demo
git clone https://github.com/SmartHarmony/JPEG-AI-OPT
cd JPEG-AI-OPT
```

### 2. Set up Python path and libraries
First, configure the Python path to align with your specific environment.
> *Note*: This instruction uses `Python 3.7` as an example. However, Python versions later than 3.7 should also be compatible.  Make sure to use the path corresponding to the Python version you are working with.  

Navigate to the `JPEG-AI-OPT/JAIOPT/repo/opencl-kernel` directory.
```bash
cd ~/Documents/GitHub/demo/JPEG-AI-OPT/repo/
```
 Open the file named `opencl_kernel_configure.bzl` and find line 40 in the file. It should look like this: 
```bash
python_bin_path = "/usr/local/bin/python3.7"
```

Replace `/usr/local/bin/python3.7` with the path to your `Python 3.7` installation.  
01_Locate_Python_Bin_Path  
![01_Locate_Python_Bin_Path.jpg](/images/01_Locate_Python_Bin_Path.jpg)  

To find your Python path, run the following command in your terminal:  
```bash
whereis python
```
02_whereis_Python  
![02_whereis_Python.jpg](/images/02_whereis_Python.jpg)  
  
After configuring your custom Python path, the next step is to install the required libraries.  
Navigate to the `JAIOPT` directory.  
```
cd ~/Documents/GitHub/demo/JPEG-AI-OPT/JAIOPT/
```
There, you can install the necessary Python libraries by executing the following command:
```bash
pip install -r requirements.txt
```  
03_Install_Python_Libraries
![03_Install_Python_Libraries.jpg](/images/03_Install_Python_Libraries.jpg)  
04_Install_Python_Libraries_Done  
![04_Install_Python_Libraries_Done.jpg](/images/04_Install_Python_Libraries_Done.jpg)  
> *Note:* Whenever you see any importing error, you may need to install the missing package by running
> ```bash
> pip install missing_package_name
> ```

### 3. Install required build tools and packages
On **Linux**, run:
```bash
sudo apt-get install cmake gcc g++ libboost-all-dev libncurses5
```
05_Install_Linux_Libraries
![05_Install_Linux_Libraries.jpg](/images/05_Install_Linux_Libraries.jpg)  
06_Install_Linux_Libraries_Done
![06_Install_Linux_Libraries_Done.jpg](/images/06_Install_Linux_Libraries_Done.jpg)  
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
  Here is our example  
  ```bash
  export ANDROID_SDK_HOME=/home/phoenix/Android/Sdk/
  export ANDROID_NDK_HOME=/home/phoenix/Android/Sdk/ndk/20.1.5948944/
  export PATH=/home/phoenix/Android/Sdk/tools:/home/phoenix/Android/Sdk/platform:$PATH
  ```
  21_Configure_Android_Environment
  ![21_Configure_Android_Environment.jpg](/images/21_Configure_Android_Environment.jpg)  
  
  ```bash
  echo $ANDROID_SDK_HOME
  echo $ANDROID_NDK_HOME
  echo $PATH
  ```
  22_Verify_Android_Environment
  ![22_Verify_Android_Environment.jpg](/images/22_Verify_Android_Environment.jpg)  

  If you are in `cshell` enviornments, add the following lines to your `.cshrc` file:
  ```
  setenv ANDROID_SDK_HOME ~/path/to/Android/sdk
  setenv ANDROID_NDK_HOME ~/path/to/android-ndk-r17c
  setenv PATH ~/path/to/Android/sdk/tools:~/path/to/Android/sdk/platform-tools:$PATH`
  ```
  

### 5. Check your setup
Navigate to the `JAIOPT` directory.  
```
cd ~/Documents/GitHub/demo/JPEG-AI-OPT/JAIOPT/
```
Execute the following commands:  
```
bazel build --config android --config optimization //deepvan/executor:libexecutor_shared.so --config symbol_hidden --define neon=true --define openmp=true --define opencl=true --cpu=arm64-v8a
```
07_Check_Your_Setup
![07_Check_Your_Setup.jpg](/images/07_Check_Your_Setup.jpg)  
08_Check_Your_Setup_In_Progress
![08_Check_Your_Setup_In_Progress.jpg](/images/08_Check_Your_Setup_In_Progress.jpg)  
09_Check_Your_Setup_In_Progress
![09_Check_Your_Setup_In_Progress.jpg](/images/09_Check_Your_Setup_In_Progress.jpg)  
10_Check_Your_Setup_Done
![10_Check_Your_Setup_Done.jpg](/images/10_Check_Your_Setup_Done.jpg)  

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
In this part, we are using `decoder_uv` model as an example for performance measurement.

### 1. Build JAIOPT framework

Navigate to the `JAIOPT` directory.  
```
cd ~/Documents/GitHub/demo/JPEG-AI-OPT/JAIOPT/
```
Run the `build.sh` script to initiate the build process.
```bash
./build.sh
```
11_Build_Framework
![11_Build_Framework.jpg](/images/11_Build_Framework.jpg)  
12_Build_Framework_Done
![12_Build_Framework_Done.jpg](/images/12_Build_Framework_Done.jpg)  

### 2. Configuration for model

After compiling `JAIOPT` successfully, proceed with the configuration steps for the `decoder_uv` model:

Navigate to the `Verification_Models/decoder` directory  
```
cd ~/Documents/GitHub/demo/JPEG-AI-OPT/Verification_Models/decoder/
```
Locate the configuration file `decoder_uv.yml`, which contains pre-defined parameters like model input/output shape, data type, and runtime.  
Update the `model_file_path` in this file to reflect the correct path by modifying `line 6` to:
```
 model_file_path: path/to/model/decoder_uv.onnx
```
Here is our example  
```
model_file_path: /home/phoenix/Documents/GitHub/demo/JPEG-AI-OPT/Verification_Models/decoder/decoder_uv.onnx
```
13_Update_Model_File_Path
![13_Update_Model_File_Path.jpg](/images/13_Update_Model_File_Path.jpg)  
>*Note:* This example is specific to the `decoder_uv` model. For a general `.yml` template, please refer to the [config.yml](https://github.com/SmartHarmony/JPEG-AI-OPT/blob/main/config.yml).

### 3. Measure model performance  
### 3.1 Conversion  

Execute command to convert that ONNX to our internal computational graph.:  
```bash
python3.7 path/to/JAIOPT/lothar/controller.py convert --config=path/to/decoder_uv.yml --model_path=path/to/decoder_uv.onnx
```
Here is our example  
```
python3.7 /home/phoenix/Documents/GitHub/demo/JPEG-AI-OPT/JAIOPT/lothar/controller.py convert --config=/home/phoenix/Documents/GitHub/demo/JPEG-AI-OPT/Verification_Models/decoder/decoder_uv.yml --model_path=/home/phoenix/Documents/GitHub/demo/JPEG-AI-OPT/Verification_Models/decoder/decoder_uv.onnx
```
14_Convert
![14_Convert.jpg](/images/14_Convert.jpg)  
15_Convert_In_Progress
![15_Convert_In_Progress.jpg](/images/15_Convert_In_Progress.jpg)  

>*Note:* In the given command, the `decoder_uv` model is used as a representative example. For testing with a different `ONNX` model, replace `decoder_uv` with the name of your target model and update the corresponding configuration `.yml` file accordingly. Refer to the contents of the `Verification_Models` directory for examples of model configurations.

The result of conversion should be like:  
16_Convert_Done
![16_Convert_Done.jpg](/images/16_Convert_Done.jpg)  

#### 3.2 Run
  
To run the model, connect your Android phone to this computer, enable `Developer` mode, and turn on `USB Debugging` in the `Developer Options` within your phone's Settings. Afterward, run the following command in the root directory of this tool on your computer:
```bash
python3.7 path/to/JAIOPT/lothar/controller.py run --config=path/to/decoder_uv.yml --model_path=path/to/decoder_uv.onnx
```
Here is our example  
```
python3.7 /home/phoenix/Documents/GitHub/demo/JPEG-AI-OPT/JAIOPT/lothar/controller.py run --config=/home/phoenix/Documents/GitHub/demo/JPEG-AI-OPT/Verification_Models/decoder/decoder_uv.yml --model_path=/home/phoenix/Documents/GitHub/demo/JPEG-AI-OPT/Verification_Models/decoder/decoder_uv.onnx
```
17_Run
![17_Run.jpg](/images/17_Run.jpg)  
18_Run_In_Progress
![18_Run_In_Progress.jpg](/images/18_Run_In_Progress.jpg)  
>*Note:* In the given command, the `decoder_uv` model is used as a representative example. For testing with a different `ONNX` model, replace `decoder_uv` with the name of your target model and update the corresponding configuration `.yml` file accordingly. Refer to the contents of the `Verification_Models` directory for examples of model configurations.

The `decoder_uv` model will then be executed on the smartphone on some random inputs created by the script. The running result should be like:
19_Run_Success
![19_Run_Success.jpg](/images/19_Run_Successjpg.jpg)  
20_Run_Success_More_Information
![20_Run_Success_More_Information.jpg](/images/20_Run_Success_More_Information.jpg)  

## License
  Source code uses [Apache License 2.0](https://github.com/SmartHarmony/JPEG-AI-OPT/blob/main/LICENSE)

  Other materials uses [Creative Commons Attribution (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/)
