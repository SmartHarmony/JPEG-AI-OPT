# Original build command. 
#bazel build //deepvan/executor:libexecutor_shared.so --config android --cpu=arm64-v8a --define neon=true --define openmp=true --define opencl=true --define quantize=false --define hexagon=false --define hta=false --config optimization --config symbol_hidden

# The build command with memory profiling support. It is important to add "memprof=true" so that the profiling code will be compiled. 

bazel build //deepvan/executor:libexecutor_shared.so --config android --cpu=arm64-v8a --define neon=true --define openmp=true --define opencl=true --define memprof=true --define quantize=false --define hexagon=false --define hta=false --config optimization --config symbol_hidden
