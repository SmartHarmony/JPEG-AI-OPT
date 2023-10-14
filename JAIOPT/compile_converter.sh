bazel build //lothar:converter

# The command_helper.py use `sh` and absolute path to call converter.py. Compile the lothar directory as  
# a whole python pakage is not feasible. So we compile some bazel exported subdirectories or python files
# seperately. 

# cp -f bazel-bin/deepvan/proto/deepvan_pb2.py deepvan/proto/
# cp -f bazel-bin/third_party/caffe/caffe_pb2.py third_party/caffe/

# nuitka_dir=nuitka3-bin/lothar
# converter_dist=$nuitka_dir/converter.dist
# converter_bin_path=$converter_dist/converter

# python -m nuitka --standalone --show-progress --show-memory --nofollow-import-to=tensorflow,matplotlib,multiprocessing,numpy,onnx,jinja2,markupsafe,google.protobuf --output-dir=$nuitka_dir lothar/tools/converter.py


pushd bazel-bin/lothar/converter.runfiles/deepvan/lothar
python -m nuitka --module net_converter --include-package=net_converter
rm -rf net_converter net_converter.build 
popd

pushd bazel-bin/lothar/converter.runfiles/deepvan/deepvan/proto
python -m nuitka --module deepvan_pb2.py
rm -rf deepvan_pb2.py deepvan_pb2.build
popd

rm -rf lothar/net_converter