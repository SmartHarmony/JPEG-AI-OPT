/**
 * Usage:
 * deepvan_run --model=mobi.pb \
 *          --input=input_node  \
 *          --output=output_node  \
 *          --input_shape=1,224,224,3   \
 *          --output_shape=1,224,224,2   \
 *          --input_file=input_data \
 *          --output_file=mobi.out  \
 *          --model_data_file=model_data.data \
 *          --device=GPU
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <numeric>
#include <stdint.h>
#include <thread>

#include "deepvan/compat/env.h"
#include "deepvan/compat/file_system.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/logging.h"
#include "deepvan/utils/memory.h"
#include "deepvan/utils/string_util.h"
#include "gflags/gflags.h"

namespace model_runner {

class ModelRunnerBase {
public:
  ModelRunnerBase(const std::vector<std::string> &input_names,
                  const std::vector<std::vector<int64_t>> &input_shapes,
                  const std::vector<deepvan::DataFormat> &input_data_formats,
                  const std::vector<std::string> &output_names,
                  const std::vector<std::vector<int64_t>> &output_shapes,
                  const std::vector<deepvan::DataFormat> &output_data_formats,
                  int num_rounds,
                  int num_warmup_rounds,
                  float fps,
                  bool ignore_outliers,
                  float cpu_capability)
      : input_names(input_names),
        input_shapes(input_shapes),
        input_data_formats(input_data_formats),
        output_names(output_names),
        output_shapes(output_shapes),
        output_data_formats(output_data_formats),
        num_rounds(num_rounds),
        num_warmup_rounds(num_warmup_rounds),
        fps(fps),
        ignore_outliers(ignore_outliers),
        cpu_capability(cpu_capability) {}
  virtual ~ModelRunnerBase() {}

  bool run() {
    LOG(INFO) << "start to run";
    int64_t t0 = deepvan::NowMicros();
    deepvan::VanState create_engine_status = build_model();
    int64_t t1 = deepvan::NowMicros();

    double init_millis = (t1 - t0) / 1000.0;
    LOG(INFO) << "Total init latency: " << init_millis << " ms";

    if (create_engine_status != deepvan::VanState::SUCCEED ||
        !allocate_inputs_and_outputs()) {
      LOG(INFO) << "Create model executor failed, return...";
      return false;
    }

    LOG(INFO) << "Warm up";
    double warmup_millis = 0;
    for (int i = 0; i < num_warmup_rounds; ++i) {
      double res = warmup(i);
      if (res >= std::numeric_limits<float>::max()) {
        warmup_millis = std::numeric_limits<double>::infinity();
        break;
      }
      warmup_millis += res;
    }

    std::vector<int64_t> run_durations;
    run_durations.reserve(num_rounds);

    LOG(INFO) << "Run model";
    int64_t total_run_duration = 0;
    for (int i = 0; i < num_rounds; ++i) {
      int64_t res = run(i);
      if (res == std::numeric_limits<long long>::max()) {
        total_run_duration = res;
        break;
      } else {
        total_run_duration += res;
        if (ignore_outliers) {
          run_durations.push_back(res);
        }
      }

      if (fps > 1e-6) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(static_cast<uint64_t>(fps * 1e6)));
      }
    }

    double model_run_millis = 0;
    if (total_run_duration == std::numeric_limits<long long>::max()) {
      model_run_millis = std::numeric_limits<double>::infinity();
    } else if (ignore_outliers && run_durations.size() >= 4) {
      // box plot
      auto num_runs = run_durations.size();
      std::sort(run_durations.begin(), run_durations.end());
      auto Q1 = run_durations[num_runs / 4];
      auto Q3 = run_durations[num_runs - num_runs / 4 - 1];
      auto IQR = Q3 - Q1;
      auto lower_boundary = Q1 - 1.5 * IQR;
      auto upper_boundary = Q3 + 1.5 * IQR;
      model_run_millis = 0;
      int num_outliers = 0;
      for (auto latency : run_durations) {
        if (latency >= lower_boundary && latency <= upper_boundary) {
          model_run_millis += latency;
        } else {
          ++num_outliers;
        }
      }
      model_run_millis = model_run_millis / 1000.0 / (num_runs - num_outliers);
      LOG(INFO) << "Average latency (w/o " << num_outliers
                << " outliers): " << model_run_millis << " ms, boundaries: ["
                << lower_boundary / 1000.0 << ", " << upper_boundary / 1000.0
                << "]";
    } else {
      model_run_millis = total_run_duration / 1000.0 / num_rounds;
      LOG(INFO) << "Average latency: " << model_run_millis << " ms";
    }

    // Metrics reporting tools depends on the format, keep in consistent
    printf("========================================================\n");
    printf("     capability(CPU)        init      warmup     run_avg\n");
    printf("========================================================\n");
    printf("time %15.3f %11.3f %11.3f %11.3f\n",
           cpu_capability,
           init_millis,
           warmup_millis,
           model_run_millis);
    printf("Result: {\"Init\":\"%.3f\", \"Warmup\":\"%.3f\", "
           "\"Run_avg\":\"%.3f\"}",
           init_millis,
           warmup_millis,
           model_run_millis);
    std::cout << std::endl;

    int num_outputs = output_names.size();
    for (int i = 0; i < num_outputs; ++i) {
      write_output_to_file(i);
    }

    return true;
  }

protected:
  virtual deepvan::VanState build_model() = 0;
  virtual double warmup(int round) = 0;
  virtual int64_t run(int round) = 0;

  int get_num_rounds() const { return num_rounds; }
  int get_num_warmup_rounds() const { return num_warmup_rounds; }

protected:
  const std::vector<std::string> &input_names;
  const std::vector<std::vector<int64_t>> &input_shapes;
  const std::vector<deepvan::DataFormat> &input_data_formats;
  const std::vector<std::string> &output_names;
  const std::vector<std::vector<int64_t>> &output_shapes;
  const std::vector<deepvan::DataFormat> &output_data_formats;
  std::map<std::string, deepvan::TensorWrapper> inputs;
  std::map<std::string, deepvan::TensorWrapper> outputs;

private:
  bool allocate_inputs_and_outputs();
  void write_output_to_file(int index);

private:
  const int num_rounds;
  const int num_warmup_rounds;
  const float fps;
  const bool ignore_outliers;
  const float cpu_capability;
};

ModelRunnerBase *CreateModelRunner(
    const std::string &model_name,
    const std::vector<std::string> &input_names,
    const std::vector<std::vector<int64_t>> &input_shapes,
    const std::vector<deepvan::DataFormat> &input_data_formats,
    const std::vector<std::string> &output_names,
    const std::vector<std::vector<int64_t>> &output_shapes,
    const std::vector<deepvan::DataFormat> &output_data_formats,
    float fps,
    float cpu_capability);

bool RunModel(const std::string &model_name,
              const std::vector<std::string> &input_names,
              const std::vector<std::vector<int64_t>> &input_shapes,
              const std::vector<deepvan::DataFormat> &input_data_formats,
              const std::vector<std::string> &output_names,
              const std::vector<std::vector<int64_t>> &output_shapes,
              const std::vector<deepvan::DataFormat> &output_data_formats,
              float fps,
              float cpu_capability) {
  std::unique_ptr<ModelRunnerBase> runner(CreateModelRunner(model_name,
                                                            input_names,
                                                            input_shapes,
                                                            input_data_formats,
                                                            output_names,
                                                            output_shapes,
                                                            output_data_formats,
                                                            fps,
                                                            cpu_capability));
  return runner->run();
}

} // namespace model_runner

namespace deepvan {
namespace run {

template <typename T>
void ParseShape(const std::string &str, std::vector<T> *shape) {
  std::string tmp = str;
  while (!tmp.empty()) {
    int dim = atoi(tmp.data());
    shape->push_back(dim);
    size_t next_offset = tmp.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
}

std::string FormatName(const std::string input) {
  std::string res = input;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!isalnum(res[i]))
      res[i] = '_';
  }
  return res;
}

DeviceType ParseDeviceType(const std::string &device_str) {
  if (device_str.compare("CPU") == 0) {
    return DeviceType::CPU;
  } else if (device_str.compare("GPU") == 0) {
    return DeviceType::GPU;
  } else if (device_str.compare("HEXAGON") == 0) {
    return DeviceType::HEXAGON;
  } else if (device_str.compare("HTA") == 0) {
    return DeviceType::HTA;
  } else {
    return DeviceType::CPU;
  }
}

DataFormat ParseDataFormat(const std::string &data_format_str) {
  if (data_format_str == "NHWC") {
    return DataFormat::NHWC;
  } else if (data_format_str == "NCHW") {
    return DataFormat::NCHW;
  } else if (data_format_str == "OIHW") {
    return DataFormat::OIHW;
  } else {
    return DataFormat::DF_NONE;
  }
}

DEFINE_string(model_name, "", "model name in yaml");
DEFINE_string(input_node,
              "input_node0,input_node1",
              "input nodes, separated by comma");
DEFINE_string(input_shape,
              "1,224,224,3:1,1,1,10",
              "input shapes, separated by colon and comma");
DEFINE_string(output_node,
              "output_node0,output_node1",
              "output nodes, separated by comma");
DEFINE_string(output_shape,
              "1,224,224,2:1,1,1,10",
              "output shapes, separated by colon and comma");
DEFINE_string(input_data_format, "NHWC", "input data formats, NONE|NHWC|NCHW");
DEFINE_string(output_data_format,
              "NHWC",
              "output data formats, NONE|NHWC|NCHW");
DEFINE_string(input_file,
              "",
              "input file name | input file prefix for multiple inputs.");
DEFINE_string(output_file,
              "",
              "output file name | output file prefix for multiple outputs");
// TODO@vgod: support batch validation
DEFINE_string(input_dir, "", "input directory name");
DEFINE_string(output_dir, "output", "output directory name");
DEFINE_string(opencl_binary_file, "", "compiled opencl binary file path");
DEFINE_string(opencl_parameter_file, "", "tuned OpenCL parameter file path");
DEFINE_string(model_data_file,
              "",
              "model data file name, used when EMBED_MODEL_DATA set to 0 or 2");
DEFINE_string(other_data_file, "", "other data file name");
DEFINE_string(model_file,
              "",
              "model file name, used when load deepvan model in pb");
DEFINE_string(device, "GPU", "CPU/GPU/HEXAGON");
DEFINE_int32(round, 1, "round");
DEFINE_int32(restart_round, 1, "restart round");
DEFINE_int32(warmup_round, 1, "warmup round");
DEFINE_int32(attempt_round,
             0,
             "number of attempts to retry deepvan. 0 means infinite times");
DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
DEFINE_int32(omp_num_threads, 4, "num of openmp threads");
DEFINE_int32(cpu_affinity_policy,
             1,
             "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY/"
             "3:AFFINITY_PERFORMANCE_CORE_FIRST/"
             "4:AFFINITY_EFFICIENT_CORE_FIRST/5:AFFINITY_PERFORMANCE/"
             "6:AFFINITY_BALANCE/7:AFFINITY_POWER_SAVE");
DEFINE_int32(
    affinity_policy_type,
    0,
    "type of affinity policy: 0: predfined, 1: cluster: 2: full customized");
DEFINE_int32(
    cluster_id,
    0,
    "the cpu cluster to be used, valid only when affinity policy type is 1");
DEFINE_int32(
    chunk_size,
    0,
    "the openmp sched chunk size, valid only when affinity policy type is 2");
DEFINE_int32(
    omp_sched_policy,
    0,
    "the default openmp sched kind, valid only when affinity policy type is 2");
DEFINE_string(
    cpu_ids,
    "0,1,2,3",
    "cpu ids in form of 0,1,2,3, valid only when affinity policy type is 2");

DEFINE_double(sleep_seconds, 0.0, "delay execution per iteration in seconds");
DEFINE_bool(quantized, false, "run quantized tflite model");
DEFINE_bool(xgen, false, "use XGen APIs");
DEFINE_bool(ignore_latency_outliers,
            false,
            "compute average latency without outliers");

class DeepvanModelRunner : public model_runner::ModelRunnerBase {
public:
  DeepvanModelRunner(const std::string &model_name,
                     const std::vector<std::string> &input_names,
                     const std::vector<std::vector<int64_t>> &input_shapes,
                     const std::vector<DataFormat> &input_data_formats,
                     const std::vector<std::string> &output_names,
                     const std::vector<std::vector<int64_t>> &output_shapes,
                     const std::vector<DataFormat> &output_data_formats,
                     float fps,
                     float cpu_capability)
      : model_runner::ModelRunnerBase(input_names,
                                      input_shapes,
                                      input_data_formats,
                                      output_names,
                                      output_shapes,
                                      output_data_formats,
                                      FLAGS_round,
                                      FLAGS_warmup_round,
                                      fps,
                                      FLAGS_ignore_latency_outliers,
                                      cpu_capability),
        model_name(model_name),
        config(ParseDeviceType(FLAGS_device)) {}
  virtual ~DeepvanModelRunner() {}

protected:
  deepvan::VanState build_model() override {
    DeviceType device_type = ParseDeviceType(FLAGS_device);

    VanState status;
    if (FLAGS_affinity_policy_type == 0) {
      FLAGS_cpu_affinity_policy = 1;
      FLAGS_omp_num_threads = 4;

      status = config.SetCPUThreadPolicy(
          FLAGS_omp_num_threads,
          static_cast<CPUAffinityPolicy>(FLAGS_cpu_affinity_policy));
    } else if (FLAGS_affinity_policy_type == 1) {
      status =
          config.SetCPUThreadPolicy(FLAGS_omp_num_threads, FLAGS_cluster_id);
    } else if (FLAGS_affinity_policy_type == 2) {
      std::vector<size_t> cpu_ids;
      ParseShape(FLAGS_cpu_ids, &cpu_ids);

      if (cpu_ids.empty()) {
        LOG(WARNING)
            << "full customized policy is used but cpu ids is not specified.";
        exit(EXIT_FAILURE);
      }

      status = config.SetCPUThreadPolicy(
          FLAGS_omp_num_threads,
          static_cast<SchedulePolicy>(FLAGS_omp_sched_policy),
          FLAGS_chunk_size,
          cpu_ids);
    } else {
      LOG(WARNING) << "Unsupported policy type: " << FLAGS_cpu_affinity_policy;
      exit(EXIT_FAILURE);
    }

    if (status != VanState::SUCCEED) {
      LOG(WARNING) << "Set openmp or cpu affinity failed.";
    }
#ifdef OPENCL_SUPPORT
    if (device_type == DeviceType::GPU) {
      const char *storage_path_ptr = getenv("DEEPVAN_INTERNAL_STORAGE_PATH");
      const std::string storage_path = std::string(
          storage_path_ptr == nullptr ? "/data/local/tmp/deepvan_run/interior"
                                      : storage_path_ptr);
      std::vector<std::string> opencl_binary_paths = {FLAGS_opencl_binary_file};

      gpu_context = GPUContextBuilder()
                        .SetStoragePath(storage_path)
                        .SetOpenCLBinaryPaths(opencl_binary_paths)
                        .SetOpenCLParameterPath(FLAGS_opencl_parameter_file)
                        .Finalize();

      config.SetGPUContext(gpu_context);
      config.SetGPUHints(static_cast<GPUPerfHint>(FLAGS_gpu_perf_hint),
                         static_cast<GPUPriorityHint>(FLAGS_gpu_priority_hint));
    }
#endif // OPENCL_SUPPORT

    model_graph_data =
        make_unique<deepvan::compat::ReadOnlyBufferMemoryRegion>();
    if (FLAGS_model_file != "") {
      auto fs = GetFileSystem();
      status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
                                                   &model_graph_data);
      if (status != VanState::SUCCEED) {
        LOG(FATAL) << "Failed to read file: " << FLAGS_model_file;
      }
    }

    model_weights_data =
        make_unique<deepvan::compat::ReadOnlyBufferMemoryRegion>();
    if (FLAGS_model_data_file != "") {
      auto fs = GetFileSystem();
      status = fs->NewReadOnlyMemoryRegionFromFile(
          FLAGS_model_data_file.c_str(), &model_weights_data);
      if (status != VanState::SUCCEED) {
        LOG(FATAL) << "Failed to read file: " << FLAGS_model_data_file;
      }
    }
    other_weights_data =
        make_unique<deepvan::compat::ReadOnlyBufferMemoryRegion>();
    if (FLAGS_other_data_file != "") {
      auto fs = GetFileSystem();
      status = fs->NewReadOnlyMemoryRegionFromFile(
          FLAGS_other_data_file.c_str(), &other_weights_data);
      if (status != VanState::SUCCEED) {
        LOG(FATAL) << "Failed to read file: " << FLAGS_other_data_file;
      } else {
        VLOG(3) << "Succeed to read file: " << FLAGS_other_data_file;
      }
    }

    VanState create_engine_status;

    CreatePatternEngine = [&]() -> VanState {
      return CreateModelExecutorFromProto(
          reinterpret_cast<const unsigned char *>(model_graph_data->data()),
          model_graph_data->length(),
          reinterpret_cast<const unsigned char *>(model_weights_data->data()),
          model_weights_data->length(),
          reinterpret_cast<const unsigned char *>(other_weights_data->data()),
          other_weights_data->length(),
          input_names,
          output_names,
          config,
          &engine);
    };

    CreateDenseEngine = [&]() -> VanState {
      return CreateModelExecutorFromProto(
          reinterpret_cast<const unsigned char *>(model_graph_data->data()),
          model_graph_data->length(),
          reinterpret_cast<const unsigned char *>(model_weights_data->data()),
          model_weights_data->length(),
          input_names,
          output_names,
          config,
          &engine);
    };

    int attempts = FLAGS_attempt_round;
    while (true) {
      // Create Engine
      int64_t t0 = NowMicros();
      (void)(model_name);
      if (FLAGS_other_data_file != "") {
        create_engine_status = CreatePatternEngine();
      } else {
        create_engine_status = CreateDenseEngine();
      }
      int64_t t1 = NowMicros();

      if (create_engine_status != VanState::SUCCEED) {
        LOG(ERROR) << "Create engine runtime error, retry ... errcode: "
                   << create_engine_status.information();
        if (attempts > 0) {
          if (--attempts == 0) {
            engine_failed = true;
            break;
          }
        }
      } else {
        double create_engine_millis = (t1 - t0) / 1000.0;
        LOG(INFO) << "Create Deepvan Engine latency: " << create_engine_millis
                  << " ms";
        break;
      }
    }
    return create_engine_status;
  }

  double warmup(int round) override {
    if (engine_failed) {
      return std::numeric_limits<float>::max();
    }
    double warmup_millis = std::numeric_limits<float>::max();
    VanState create_engine_status;
    int attempts = FLAGS_attempt_round;
    while (true) {
      int64_t t3 = NowMicros();
      VanState warmup_status = engine->Run(inputs, &outputs);
      if (warmup_status != VanState::SUCCEED) {
        LOG(ERROR) << "Warmup runtime error, retry ... errcode: "
                   << warmup_status.information();
        if (attempts > 0) {
          if (--attempts == 0) {
            engine_failed = true;
            break;
          }
        }
        do {
          if (FLAGS_other_data_file != "") {
            create_engine_status = CreatePatternEngine();
          } else {
            create_engine_status = CreateDenseEngine();
          }
        } while (create_engine_status != VanState::SUCCEED);
      } else {
        int64_t t4 = NowMicros();
        warmup_millis = (t4 - t3) / 1000.0;
        LOG(INFO) << "warm up run round " << round + 1
                  << ", latency: " << warmup_millis << " ms";
        break;
      }
    }
    return warmup_millis;
  }

  int64_t run(int round) override {
    if (engine_failed) {
      return std::numeric_limits<long long>::max();
    }
    std::unique_ptr<compat::Logger> info_log;
    std::unique_ptr<compat::MallocLogger> malloc_logger;
    if (FLAGS_malloc_check_cycle >= 1 &&
        round % FLAGS_malloc_check_cycle == 0) {
      info_log = LOG_PTR(INFO);
      malloc_logger = compat::Env::Default()->NewMallocLogger(
          info_log.get(), MakeString(round));
    }

    VanState create_engine_status;
    int64_t total_run_duration = std::numeric_limits<long long>::max();
    int attempts = FLAGS_attempt_round;
    while (true) {
      int64_t t0 = NowMicros();
      VanState run_status = engine->Run(inputs, &outputs);
      if (run_status != VanState::SUCCEED) {
        LOG(ERROR) << "Deepvan run model runtime error, retry ... errcode: "
                   << run_status.information();
        if (attempts > 0) {
          if (--attempts == 0) {
            engine_failed = true;
            break;
          }
        }
        do {
          if (FLAGS_other_data_file != "") {
            create_engine_status = CreatePatternEngine();
          } else {
            create_engine_status = CreateDenseEngine();
          }
        } while (create_engine_status != VanState::SUCCEED);
      } else {
        int64_t t1 = NowMicros();
        total_run_duration = (t1 - t0);
        break;
      }
    }

    return total_run_duration;
  }

private:
  const std::string &model_name;
  std::shared_ptr<deepvan::ModelExecutor> engine;
  ModelExecutionConfig config;
  std::shared_ptr<GPUContext> gpu_context;
  std::unique_ptr<deepvan::compat::ReadOnlyMemoryRegion> model_graph_data;
  std::unique_ptr<deepvan::compat::ReadOnlyMemoryRegion> model_weights_data;
  std::unique_ptr<deepvan::compat::ReadOnlyMemoryRegion> other_weights_data;
  std::function<VanState()> CreatePatternEngine;
  std::function<VanState()> CreateDenseEngine;
  bool engine_failed = false;
};

int Main(int argc, char **argv) {
  std::string usage =
      "deepvan run model\nusage: " + std::string(argv[0]) + " [flags]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "model name: " << FLAGS_model_name;
  LOG(INFO) << "input node: " << FLAGS_input_node;
  LOG(INFO) << "input shape: " << FLAGS_input_shape;
  LOG(INFO) << "output node: " << FLAGS_output_node;
  LOG(INFO) << "output shape: " << FLAGS_output_shape;
  LOG(INFO) << "input_file: " << FLAGS_input_file;
  LOG(INFO) << "output_file: " << FLAGS_output_file;
  LOG(INFO) << "model_data_file: " << FLAGS_model_data_file;
  LOG(INFO) << "model_file: " << FLAGS_model_file;
  LOG(INFO) << "device: " << FLAGS_device;
  LOG(INFO) << "round: " << FLAGS_round;
  LOG(INFO) << "restart_round: " << FLAGS_restart_round;
  LOG(INFO) << "gpu_perf_hint: " << FLAGS_gpu_perf_hint;
  LOG(INFO) << "gpu_priority_hint: " << FLAGS_gpu_priority_hint;
  LOG(INFO) << "omp_num_threads: " << FLAGS_omp_num_threads;
  LOG(INFO) << "cpu_affinity_policy: " << FLAGS_cpu_affinity_policy;
  LOG(INFO) << "affinity_policy_type: " << FLAGS_affinity_policy_type;
  LOG(INFO) << "cpu_ids: " << FLAGS_cpu_ids;
  LOG(INFO) << "cluster_id: " << FLAGS_cluster_id;

  std::vector<std::string> input_names = Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = Split(FLAGS_output_node, ',');
  std::vector<std::string> input_shapes = Split(FLAGS_input_shape, ':');
  std::vector<std::string> output_shapes = Split(FLAGS_output_shape, ':');

  const size_t input_count = input_shapes.size();
  const size_t output_count = output_shapes.size();
  std::vector<std::vector<int64_t>> input_shape_vec(input_count);
  std::vector<std::vector<int64_t>> output_shape_vec(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    ParseShape(input_shapes[i], &input_shape_vec[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    ParseShape(output_shapes[i], &output_shape_vec[i]);
  }
  std::vector<std::string> raw_input_data_formats =
      Split(FLAGS_input_data_format, ',');
  std::vector<std::string> raw_output_data_formats =
      Split(FLAGS_output_data_format, ',');
  std::vector<DataFormat> input_data_formats(input_count);
  std::vector<DataFormat> output_data_formats(output_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_data_formats[i] = ParseDataFormat(raw_input_data_formats[i]);
  }
  for (size_t i = 0; i < output_count; ++i) {
    output_data_formats[i] = ParseDataFormat(raw_output_data_formats[i]);
  }

  float cpu_float32_performance = 10.0;
  bool ret = false;
  for (int i = 0; i < FLAGS_restart_round; ++i) {
    VLOG(0) << "restart round " << i;
    ret = model_runner::RunModel(FLAGS_model_name,
                                 input_names,
                                 input_shape_vec,
                                 input_data_formats,
                                 output_names,
                                 output_shape_vec,
                                 output_data_formats,
                                 FLAGS_sleep_seconds,
                                 cpu_float32_performance);
  }
  if (ret) {
    return 0;
  }
  return -1;
}

} // namespace run
} // namespace deepvan

namespace model_runner {

bool ModelRunnerBase::allocate_inputs_and_outputs() {
  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();

  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    // only support float and int32, use char for generalization
    // sizeof(int) == 4, sizeof(float) == 4
    int64_t input_size = std::accumulate(input_shapes[i].begin(),
                                         input_shapes[i].end(),
                                         4,
                                         std::multiplies<int64_t>());
    char *padded_in = reinterpret_cast<char *>(memalign(64, input_size));
    auto buffer_in =
        std::shared_ptr<char>(padded_in, std::default_delete<char[]>());
    // load input
    std::ifstream in_file(deepvan::run::FLAGS_input_file + "_" +
                              deepvan::run::FormatName(input_names[i]),
                          std::ios::in | std::ios::binary);
    if (in_file.is_open()) {
      in_file.read(buffer_in.get(), input_size);
      in_file.close();
    } else {
      LOG(INFO) << "Open input file failed";
      return false;
    }
    inputs[input_names[i]] = deepvan::TensorWrapper(
        input_shapes[i], buffer_in, input_data_formats[i]);
  }

  for (size_t i = 0; i < output_count; ++i) {
    // only support float and int32, use char for generalization
    int64_t output_size = std::accumulate(output_shapes[i].begin(),
                                          output_shapes[i].end(),
                                          4,
                                          std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<char>(new char[output_size],
                                            std::default_delete<char[]>());
    outputs[output_names[i]] = deepvan::TensorWrapper(
        output_shapes[i], buffer_out, output_data_formats[i]);
  }

  return true;
}

void ModelRunnerBase::write_output_to_file(int index) {
  std::string output_name = deepvan::run::FLAGS_output_file + "_" +
                            deepvan::run::FormatName(output_names[index]);
  std::ofstream out_file(output_name, std::ios::binary);
  // only support float and int32
  int64_t output_size = std::accumulate(output_shapes[index].begin(),
                                        output_shapes[index].end(),
                                        4,
                                        std::multiplies<int64_t>());
  out_file.write(outputs[output_names[index]].data<char>().get(), output_size);
  out_file.flush();
  out_file.close();
  VLOG(deepvan::INFO) << "Write output file " << output_name << " with size "
                      << output_size << " done.";
}

ModelRunnerBase *CreateModelRunner(
    const std::string &model_name,
    const std::vector<std::string> &input_names,
    const std::vector<std::vector<int64_t>> &input_shapes,
    const std::vector<deepvan::DataFormat> &input_data_formats,
    const std::vector<std::string> &output_names,
    const std::vector<std::vector<int64_t>> &output_shapes,
    const std::vector<deepvan::DataFormat> &output_data_formats,
    float fps,
    float cpu_capability) {

  LOG(INFO) << "create deepvan model runner";
  return new deepvan::run::DeepvanModelRunner(model_name,
                                              input_names,
                                              input_shapes,
                                              input_data_formats,
                                              output_names,
                                              output_shapes,
                                              output_data_formats,
                                              fps,
                                              cpu_capability);
}

} // namespace model_runner

int main(int argc, char **argv) { return deepvan::run::Main(argc, argv); }
