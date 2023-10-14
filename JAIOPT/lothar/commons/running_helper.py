import os
import sys
import copy
import subprocess
import time

import six
import sh

from lothar.commons import common
from lothar.commons.common import *
from lothar.commons import command_helper


WORKING_DIR = "/".join(__file__.split("/")[0:-3])


class DeviceWrapper:
    allow_scheme = ('ssh', 'adb')

    def __init__(self, device_dict):
        diff = set(device_dict.keys()) - set(YAMLKeyword.__dict__.keys())
        if len(diff) > 0:
            six.print_('Wrong key detected: ')
            six.print_(diff)
            raise KeyError(str(diff))
        self.__dict__.update(device_dict)
        if self.system == SystemType.android:
            self.data_dir = PHONE_DATA_DIR
            self.interior_dir = self.data_dir + '/interior'
        elif self.system == SystemType.arm_linux:
            try:
                sh.ssh('-q', '{}@{}'.format(self.username, self.address),
                       'exit')
            except sh.ErrorReturnCode as e:
                six.print_('device connect failed, '
                           'please check your authentication')
                raise e
            self.data_dir = DEVICE_DATA_DIR
            self.interior_dir = self.data_dir + '/interior'
        elif self.system == SystemType.host:
            self.data_dir = DEVICE_DATA_DIR
            self.interior_dir = self.data_dir + '/interior'

    ##################
    #  internal use  #
    ##################

    def exec_command(self, command, *args, **kwargs):
        if self.system == SystemType.android:
            sh.adb('-s', self.address, 'shell', command, *args, **kwargs)
        elif self.system == SystemType.arm_linux:
            sh.ssh('{}@{}'.format(self.username, self.address),
                   command, *args, **kwargs)

    #####################
    #  public interface #
    #####################

    def is_lock(self):
        return command_helper.is_device_locked(self.address)

    def lock(self):
        return command_helper.device_lock(self.address)

    def clear_data_dir(self):
        if self.system == SystemType.android:
            # command_helper.delete_files(
            #     ['cmd_file-', 'debug-', 'model_out_', 'model_input_'],
            #     self.data_dir, self.address)
            command_helper.clear_phone_data_dir(self.address, PHONE_DATA_DIR)
            print("Clear data on /data/tmp/local/deepvan_run/")
        elif self.system == SystemType.arm_linux:
            self.exec_command('rm -rf {}'.format(self.data_dir))

    def pull_from_data_dir(self, filename, dst_path):
        if self.system == SystemType.android:
            self.pull(PHONE_DATA_DIR, filename, dst_path)
        elif self.system == SystemType.arm_linux:
            self.pull(DEVICE_DATA_DIR, filename, dst_path)

    def create_internal_storage_dir(self):
        internal_storage_dir = '{}/interior/'.format(self.data_dir)
        if self.system == SystemType.android:
            command_helper.create_internal_storage_dir(self.address,
                                                       internal_storage_dir)
        elif self.system == SystemType.arm_linux:
            self.exec_command('mkdir -p {}'.format(internal_storage_dir))
        return internal_storage_dir

    def rm(self, file):
        if self.system == SystemType.android:
            sh.adb('-s', self.address, 'shell', 'rm', '-rf', file, _fg=True)
        elif self.system == SystemType.arm_linux:
            self.exec_command('rm -rf {}'.format(file), _fg=True)

    def push(self, src_path, dst_path):
        CONDITIONS(os.path.exists(src_path), "Device",
                   '{} not found'.format(src_path))
        six.print_("Push %s to %s" % (src_path, dst_path))
        if self.system == SystemType.android:
            command_helper.adb_push(src_path, dst_path, self.address)
        elif self.system == SystemType.arm_linux:
            try:
                sh.scp(src_path, '{}@{}:{}'.format(self.username,
                                                   self.address,
                                                   dst_path))
            except sh.ErrorReturnCode_1 as e:
                six.print_('Push Failed !', e, file=sys.stderr)
                raise e

    def pull(self, src_path, file_name, dst_path='.'):
        if not os.path.exists(dst_path):
            sh.mkdir("-p", dst_path)
        src_file = "%s/%s" % (src_path, file_name)
        dst_file = "%s/%s" % (dst_path, file_name)
        if os.path.exists(dst_file):
            sh.rm('-f', dst_file)
        six.print_("Pull %s to %s" % (src_file, dst_path))
        if self.system == SystemType.android:
            command_helper.adb_pull(
                src_file, dst_file, self.address)
        elif self.system == SystemType.arm_linux:
            try:
                sh.scp('-r', '%s@%s:%s' % (self.username,
                                           self.address,
                                           src_file),
                       dst_file)
            except sh.ErrorReturnCode_1 as e:
                six.print_("Pull Failed !", file=sys.stderr)
                raise e

    def push_cmd(self, cmd, extra_args, prefix, model_tag, suffix):
        cmd.extend(extra_args)
        cmd = ' '.join(cmd)
        cmd_file_name = f'{prefix}-{model_tag}-{suffix}'
        cmd_file = f'{self.data_dir}/{cmd_file_name}'
        tmp_cmd_file = f'/tmp/{cmd_file_name}'
        with open(tmp_cmd_file, 'w') as file:
            file.write(cmd)
        self.push(tmp_cmd_file, cmd_file)
        os.remove(tmp_cmd_file)
        return cmd_file

    def tuning_run(self,
                   abi,
                   target_dir,
                   target_name,
                   vlog_level,
                   model_output_dir,
                   input_nodes,
                   output_nodes,
                   input_shapes,
                   input_data_formats,
                   output_shapes,
                   output_data_formats,
                   deepvan_model_dir,
                   model_tag,
                   device_type,
                   running_round,
                   restart_round,
                   limit_opencl_kernel_time,
                   tuning,
                   out_of_range_check,
                   opencl_binary_file,
                   opencl_parameter_file,
                   executor_dynamic_library_path,
                   omp_num_threads=-1,
                   cpu_affinity_policy=1,
                   gpu_perf_hint=3,
                   gpu_priority_hint=3,
                   input_file_name='model_input',
                   output_file_name='model_out',
                   input_dir="",
                   output_dir="",
                   address_sanitizer=False,
                   link_dynamic=False,
                   quantize_stat=False,
                   layers_validate_file="",
                   pruning_type=PruningType.DENSE,
                   perf=False):
        six.print_("* Run '%s' with round=%s, restart_round=%s, tuning=%s, "
                   "out_of_range_check=%s, omp_num_threads=%s, "
                   "cpu_affinity_policy=%s, gpu_perf_hint=%s, "
                   "gpu_priority_hint=%s, perf=%s" %
                   (model_tag, running_round, restart_round, str(tuning),
                    str(out_of_range_check), omp_num_threads,
                    cpu_affinity_policy, gpu_perf_hint, gpu_priority_hint, perf))
        deepvan_model_path = layers_validate_file if layers_validate_file \
            else "%s/%s.pb" % (deepvan_model_dir, model_tag)

        model_data_file = ""
        other_data_file = ""
        if self.system == SystemType.host:
            model_data_file = "%s/%s.data" % (deepvan_model_dir, model_tag)
        else:
            model_data_file = "%s/%s.data" % (self.data_dir, model_tag)

        if self.system == SystemType.host:
            executor_dynamic_lib_path = \
                os.path.dirname(executor_dynamic_library_path)
            p = subprocess.Popen(
                [
                    "env",
                    "ASAN_OPTIONS=detect_leaks=1",
                    "LD_LIBRARY_PATH=%s" % executor_dynamic_lib_path,
                    "DEEPVAN_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
                    "DEEPVAN_LOG_TENSOR_RANGE=%d" % (
                        1 if quantize_stat else 0),
                    "%s/%s" % (target_dir, target_name),
                    "--model_name=%s" % model_tag,
                    "--input_node=\"%s\"" % ",".join(input_nodes),
                    "--output_node=\"%s\"" % ",".join(output_nodes),
                    "--input_shape=%s" % ":".join(input_shapes),
                    "--output_shape=%s" % ":".join(output_shapes),
                    "--input_data_format=%s" % ",".join(input_data_formats),
                    "--output_data_format=%s" % ",".join(output_data_formats),
                    "--input_file=%s/%s" % (model_output_dir,
                                            input_file_name),
                    "--output_file=%s/%s" % (model_output_dir,
                                             output_file_name),
                    "--input_dir=%s" % input_dir,
                    "--output_dir=%s" % output_dir,
                    "--model_data_file=%s" % model_data_file,
                    "--other_data_file=%s" % other_data_file,
                    "--device=%s" % device_type,
                    "--round=%s" % running_round,
                    "--restart_round=%s" % restart_round,
                    "--omp_num_threads=%s" % omp_num_threads,
                    "--cpu_affinity_policy=%s" % cpu_affinity_policy,
                    "--gpu_perf_hint=%s" % gpu_perf_hint,
                    "--gpu_priority_hint=%s" % gpu_priority_hint,
                    "--model_file=%s" % deepvan_model_path,
                ],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE)
            out, err = p.communicate()
            self.stdout = err + out
            six.print_(self.stdout.decode('UTF-8'))
            six.print_("Running finished!\n")
        elif self.system in [SystemType.android, SystemType.arm_linux]:
            model_data_path = "%s/%s.data" % (deepvan_model_dir, model_tag)
            model_data_exst = command_helper.compare_md5(
                model_data_path, "%s/%s.data" % (self.data_dir, model_tag), self.address)
            command_helper.delete_files(
                ['cmd_file-', 'debug-', 'model_out_', 'model_input_'],
                self.data_dir, self.address)
            if not model_data_exst:
                self.rm(self.data_dir)
                self.exec_command('mkdir -p {}'.format(self.data_dir))
            else:
                self.exec_command('rm -rf !(%s)' % "%s/%s.data" %
                                  (self.data_dir, model_tag))
            internal_storage_dir = self.create_internal_storage_dir()

            for input_name in input_nodes:
                formatted_name = common.formatted_file_name(input_file_name,
                                                            input_name)
                self.push("%s/%s" % (model_output_dir, formatted_name),
                          self.data_dir)
            if self.system == SystemType.android and address_sanitizer:
                self.push(command_helper.find_asan_rt_library(abi),
                          self.data_dir)

            CONDITIONS(os.path.exists(model_data_path), "Device",
                       'model data file not found,'
                       ' please convert model first')
            if not model_data_exst:
                self.push(model_data_path, self.data_dir)
            paten_data_path = "%s/%s.ptd" % (deepvan_model_dir, model_tag)
            if os.path.exists(paten_data_path):
                self.push(paten_data_path, self.data_dir)
            csr_data_path = "%s/%s.csr" % (deepvan_model_dir, model_tag)
            if os.path.exists(csr_data_path):
                self.push(csr_data_path, self.data_dir)
            slice_data_path = "%s/%s.slice" % (deepvan_model_dir, model_tag)
            if os.path.exists(slice_data_path):
                self.push(slice_data_path, self.data_dir)

            if device_type == common.DeviceType.GPU:
                if os.path.exists(opencl_binary_file):
                    self.push(opencl_binary_file, self.data_dir)
                if os.path.exists(opencl_parameter_file):
                    self.push(opencl_parameter_file, self.data_dir)

            if self.system == SystemType.android \
                    and device_type == common.DeviceType.HEXAGON:
                self.push(
                    "third_party/nnlib/%s/libhexagon_controller.so" % abi,
                    self.data_dir)

            deepcan_model_phone_path = ""
            deepcan_model_phone_path = "%s/%s.pb" % (self.data_dir,
                                                     model_tag)

            other_model_path = "%s/%s.tflite" % (deepvan_model_dir, model_tag)
            other_model_phone_path = "%s/%s.tflite" % (self.data_dir,
                                                       model_tag)
            self.push(deepvan_model_path, deepcan_model_phone_path)
            if os.path.exists(other_model_path):
                self.push(other_model_path, other_model_phone_path)

            if link_dynamic:
                self.push(executor_dynamic_library_path, self.data_dir)
                if self.system == SystemType.android:
                    command_helper.push_depended_so_libs(
                        executor_dynamic_library_path, abi, self.data_dir,
                        self.address)
            self.push("%s/%s" % (target_dir, target_name), self.data_dir)

            stdout_buff = []
            process_output = command_helper.make_output_processor(stdout_buff)
            cmd = [
                "LD_LIBRARY_PATH=%s" % self.data_dir,
                "DEEPVAN_TUNING=%s" % int(tuning),
                "OUT_OF_RANGE_CHECK=%s" % int(out_of_range_check),
                "DEEPVAN_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
                "DEEPVAN_RUN_PARAMETER_PATH=%s/deepvan_run.config" % self.data_dir,
                "DEEPVAN_INTERNAL_STORAGE_PATH=%s" % internal_storage_dir,
                "DEEPVAN_LIMIT_OPENCL_KERNEL_TIME=%s" % limit_opencl_kernel_time,
                "DEEPVAN_LOG_TENSOR_RANGE=%d" % (1 if quantize_stat else 0),
            ]
            if self.system == SystemType.android and address_sanitizer:
                cmd.extend([
                    "LD_PRELOAD=%s/%s" %
                    (self.data_dir,
                     command_helper.asan_rt_library_names(abi))
                ])
            if perf:
                if len(os.environ.get('SIMPLEPERF', "")) > 0:
                    self.push(os.environ['SIMPLEPERF'], self.data_dir)
                else:
                    self.push(command_helper.find_simpleperf_library(
                        abi), self.data_dir)
                simpleperf_cmd = '/data/local/tmp/simpleperf'
                perf_params = [
                    simpleperf_cmd,
                    'stat',
                    '--group',
                    'raw-l1-dcache,raw-l1-dcache-refill',
                    '--group',
                    'raw-l2-dcache,raw-l2-dcache-refill',
                    '--group',
                    'raw-l3-dcache,raw-l3-dcache-refill',
                    '--group',
                    'raw-l1-dtlb,raw-l1-dtlb-refill',
                    '--group',
                    'raw-l2-dtlb,raw-l2-dtlb-refill',
                ]
                cmd.extend(perf_params)
            target_with_params = [
                "%s/%s" % (self.data_dir, target_name),
                "--model_name=%s" % model_tag,
                "--input_node=\"%s\"" % ",".join(input_nodes),
                "--output_node=\"%s\"" % ",".join(output_nodes),
                "--input_shape=%s" % ":".join(input_shapes),
                "--output_shape=%s" % ":".join(output_shapes),
                "--input_data_format=%s" % ",".join(input_data_formats),
                "--output_data_format=%s" % ",".join(output_data_formats),
                "--input_file=%s/%s" % (self.data_dir, input_file_name),
                "--output_file=%s/%s" % (self.data_dir, output_file_name),
                "--input_dir=%s" % input_dir,
                "--output_dir=%s" % output_dir,
                "--model_data_file=%s" % model_data_file,
                "--other_data_file=%s" % other_data_file,
                "--device=%s" % device_type,
                "--round=%s" % running_round,
                "--restart_round=%s" % restart_round,
                "--omp_num_threads=%s" % omp_num_threads,
                "--cpu_affinity_policy=%s" % cpu_affinity_policy,
                "--gpu_perf_hint=%s" % gpu_perf_hint,
                "--gpu_priority_hint=%s" % gpu_priority_hint,
                "--opencl_binary_file=%s/%s" %
                (self.data_dir, os.path.basename(opencl_binary_file)),
                "--opencl_parameter_file=%s/%s" %
                (self.data_dir, os.path.basename(opencl_parameter_file)),
            ]
            deepvan_target_with_param = [
                "--model_file=%s" % deepcan_model_phone_path,
            ]
            other_target_with_param = [
                "--model_file=%s" % other_model_phone_path,
            ]
            cmd.extend(target_with_params)
            time_stamp = str(time.time())
            if os.path.exists(other_model_path):
                other_cmd = copy.deepcopy(cmd)
                self.push_cmd(other_cmd, other_target_with_param,
                              'cmd_file-other', model_tag, time_stamp)
            cmd.extend(deepvan_target_with_param)
            cmd_file = self.push_cmd(cmd, deepvan_target_with_param,
                                     'cmd_file', model_tag, time_stamp)
            debug_cmd_file_name = "%s-%s-%s" % ('debug',
                                                model_tag,
                                                time_stamp)
            debug_cmd_file = "%s/%s" % (self.data_dir, debug_cmd_file_name)
            tmp_debug_cmd_file = "%s/%s" % ('/tmp', debug_cmd_file_name)
            debug_cmd = "/data/local/tmp/gdbserver :10010 " + \
                ' '.join(target_with_params)
            with open(tmp_debug_cmd_file, 'w') as file:
                file.write(debug_cmd)
            self.push(tmp_debug_cmd_file, debug_cmd_file)
            os.remove(tmp_debug_cmd_file)
            # exec running command
            self.exec_command('sh {}'.format(cmd_file),
                              _tty_in=True,
                              _out=process_output,
                              _err_to_out=True)
            self.stdout = "".join(stdout_buff)
            if not command_helper.stdout_success(self.stdout):
                common.DLogger.error("Deepvan Run", "Deepvan run failed.")

            six.print_("Running finished!\n")
        else:
            six.print_('Unsupported system %s' % self.system, file=sys.stderr)
            raise Exception('Wrong device')

        return self.stdout

    def tuning(self, library_name, model_name, model_config,
               target_abi, deepvan_lib_type):
        six.print_('* Tuning, it may take some time')
        build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
        deepvan_run_name = DEEPVAN_RUN_STATIC_NAME
        link_dynamic = False
        if deepvan_lib_type == DEEPVANLibType.dynamic:
            deepvan_run_name = DEEPVAN_RUN_DYNAMIC_NAME
            link_dynamic = True

        # build for specified soc
        # device_wrapper = DeviceWrapper(device)

        model_output_base_dir, model_output_dir, deepvan_model_dir = \
            get_build_model_dirs(
                library_name, model_name, target_abi, self,
                model_config[YAMLKeyword.model_file_path])

        self.clear_data_dir()

        subgraphs = model_config[YAMLKeyword.subgraphs]
        # generate input data
        command_helper.gen_random_input(
            model_output_dir,
            subgraphs[0][YAMLKeyword.input_tensors],
            subgraphs[0][YAMLKeyword.input_shapes],
            subgraphs[0][YAMLKeyword.validation_inputs_data],
            input_ranges=subgraphs[0][YAMLKeyword.input_ranges],
            input_data_types=subgraphs[0][YAMLKeyword.input_data_types]
        )

        self.tuning_run(
            abi=target_abi,
            target_dir=build_tmp_binary_dir,
            target_name=deepvan_run_name,
            vlog_level=0,
            model_output_dir=model_output_dir,
            input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
            output_nodes=subgraphs[0][YAMLKeyword.output_tensors],
            input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
            output_shapes=subgraphs[0][YAMLKeyword.output_shapes],
            input_data_formats=subgraphs[0][YAMLKeyword.input_data_formats],
            output_data_formats=subgraphs[0][YAMLKeyword.output_data_formats],
            deepvan_model_dir=deepvan_model_dir,
            model_tag=model_name,
            device_type=DeviceType.GPU,
            running_round=0,
            restart_round=1,
            limit_opencl_kernel_time=model_config[
                YAMLKeyword.limit_opencl_kernel_time],
            tuning=True,
            out_of_range_check=False,
            opencl_binary_file='',
            opencl_parameter_file='',
            executor_dynamic_library_path=LIBDEEPVAN_DYNAMIC_PATH,
            link_dynamic=link_dynamic,
        )

        # pull opencl library
        self.pull(self.interior_dir, CL_COMPILED_BINARY_FILE_NAME,
                  '{}/{}'.format(model_output_dir,
                                 BUILD_TMP_OPENCL_BIN_DIR))

        # pull opencl parameter
        self.pull_from_data_dir(CL_TUNED_PARAMETER_FILE_NAME,
                                '{}/{}'.format(model_output_dir,
                                               BUILD_TMP_OPENCL_BIN_DIR))

        six.print_('Tuning done! \n')

    @staticmethod
    def get_layers(model_dir, model_name, layers):
        command_helper.bazel_build_common("//lothar:layers_validate")

        model_file = "%s/%s.pb" % (model_dir, model_name)
        output_dir = "%s/output_models/" % model_dir
        if os.path.exists(output_dir):
            sh.rm('-rf', output_dir)
        os.makedirs(output_dir)
        sh.python3("bazel-bin/lothar/layers_validate",
                   "-u",
                   "--model_file=%s" % model_file,
                   "--output_dir=%s" % output_dir,
                   "--layers=%s" % layers,
                   _fg=True)

        output_configs_path = output_dir + "outputs.yml"
        with open(output_configs_path) as f:
            output_configs = yaml.load(f)
        output_configs = output_configs[YAMLKeyword.subgraphs]

        return output_configs

    def run_specify_abi(self, flags, configs, target_abi, pruning_type=PruningType.DENSE):
        if target_abi not in self.target_abis:
            six.print_('The device %s with soc %s do not support the abi %s' %
                       (self.device_name, self.target_socs, target_abi))
            return
        library_name = configs[YAMLKeyword.library_name]
        deepvan_lib_type = flags.deepvan_lib_type
        build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)

        # get target name for run
        if deepvan_lib_type == DEEPVANLibType.static:
            target_name = DEEPVAN_RUN_STATIC_NAME
        else:
            target_name = DEEPVAN_RUN_DYNAMIC_NAME
        link_dynamic = deepvan_lib_type == DEEPVANLibType.dynamic
        model_output_dirs = []

        for model_name in configs[YAMLKeyword.models]:
            check_model_converted(library_name, model_name, target_abi)
            if target_abi != ABIType.host:
                self.clear_data_dir()
            DLogger.header(
                StringFormatter.block(
                    'Run model {} on {}'.format(model_name, self.device_name)))

            model_config = configs[YAMLKeyword.models][model_name]
            model_runtime = model_config[YAMLKeyword.runtime]
            subgraphs = model_config[YAMLKeyword.subgraphs]

            model_output_base_dir, model_output_dir, deepvan_model_dir = \
                get_build_model_dirs(
                    library_name, model_name, target_abi, self,
                    model_config[YAMLKeyword.model_file_path])

            # clear temp model output dir
            if os.path.exists(model_output_dir):
                sh.rm('-rf', model_output_dir)
            os.makedirs(model_output_dir)

            is_tuned = False
            model_opencl_output_bin_path = ''
            model_opencl_parameter_path = ''
            if not flags.address_sanitizer \
                    and target_abi != ABIType.host \
                    and (configs[YAMLKeyword.target_socs]
                         or flags.target_socs) \
                    and self.target_socs \
                    and model_runtime in [RuntimeType.gpu,
                                          RuntimeType.cpu_gpu] \
                    and not flags.disable_tuning:
                self.tuning(library_name, model_name, model_config,
                            target_abi, deepvan_lib_type)
                model_output_dirs.append(model_output_dir)
                model_opencl_output_bin_path = \
                    '{}/{}/{}'.format(model_output_dir,
                                      BUILD_TMP_OPENCL_BIN_DIR,
                                      CL_COMPILED_BINARY_FILE_NAME)
                model_opencl_parameter_path = \
                    '{}/{}/{}'.format(model_output_dir,
                                      BUILD_TMP_OPENCL_BIN_DIR,
                                      CL_TUNED_PARAMETER_FILE_NAME)
                self.clear_data_dir()
                is_tuned = True
            elif target_abi != ABIType.host and self.target_socs:
                # TODO @vgod
                model_opencl_output_bin_path = ''
                model_opencl_parameter_path = ''
            command_helper.gen_random_input(
                model_output_dir,
                subgraphs[0][YAMLKeyword.input_tensors],
                subgraphs[0][YAMLKeyword.input_shapes],
                subgraphs[0][YAMLKeyword.validation_inputs_data],
                input_ranges=subgraphs[0][YAMLKeyword.input_ranges],
                input_data_types=subgraphs[0][YAMLKeyword.input_data_types]
            )
            runtime_list = []
            if target_abi == ABIType.host:
                runtime_list.append(RuntimeType.cpu)
            elif model_runtime == RuntimeType.cpu_gpu:
                runtime_list.extend([RuntimeType.cpu, RuntimeType.gpu])
            else:
                runtime_list.append(model_runtime)
            for runtime in runtime_list:
                device_type = parse_device_type(runtime)
                # run for specified soc
                if not subgraphs[0][YAMLKeyword.check_tensors]:
                    output_nodes = subgraphs[0][YAMLKeyword.output_tensors]
                    output_shapes = subgraphs[0][YAMLKeyword.output_shapes]
                else:
                    output_nodes = subgraphs[0][YAMLKeyword.check_tensors]
                    output_shapes = subgraphs[0][YAMLKeyword.check_shapes]
                output_configs = []
                log_file = ""
                if flags.layers != "-1":
                    output_configs = self.get_layers(deepvan_model_dir,
                                                     model_name,
                                                     flags.layers)
                    log_dir = deepvan_model_dir + "/" + runtime
                    if os.path.exists(log_dir):
                        sh.rm('-rf', log_dir)
                    os.makedirs(log_dir)
                    log_file = log_dir + "/log.csv"
                model_path = "%s/%s.pb" % (deepvan_model_dir, model_name)
                output_config = {YAMLKeyword.model_file_path: model_path,
                                 YAMLKeyword.output_tensors: output_nodes,
                                 YAMLKeyword.output_shapes: output_shapes}
                output_configs.append(output_config)
                for output_config in output_configs:
                    run_output = self.tuning_run(
                        abi=target_abi,
                        target_dir=build_tmp_binary_dir,
                        target_name=target_name,
                        vlog_level=flags.vlog_level,
                        model_output_dir=model_output_dir,
                        input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
                        output_nodes=output_config[
                            YAMLKeyword.output_tensors],
                        input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
                        output_shapes=output_config[YAMLKeyword.output_shapes],
                        input_data_formats=subgraphs[0][
                            YAMLKeyword.input_data_formats],
                        output_data_formats=subgraphs[0][
                            YAMLKeyword.output_data_formats],
                        deepvan_model_dir=deepvan_model_dir,
                        model_tag=model_name,
                        device_type=device_type,
                        running_round=flags.round,
                        restart_round=flags.restart_round,
                        limit_opencl_kernel_time=model_config[
                            YAMLKeyword.limit_opencl_kernel_time],
                        tuning=False,
                        out_of_range_check=flags.gpu_out_of_range_check,
                        omp_num_threads=flags.omp_num_threads,
                        cpu_affinity_policy=flags.cpu_affinity_policy,
                        gpu_perf_hint=flags.gpu_perf_hint,
                        gpu_priority_hint=flags.gpu_priority_hint,
                        address_sanitizer=flags.address_sanitizer,
                        opencl_binary_file=model_opencl_output_bin_path,
                        opencl_parameter_file=model_opencl_parameter_path,
                        executor_dynamic_library_path=LIBDEEPVAN_DYNAMIC_PATH,
                        link_dynamic=link_dynamic,
                        quantize_stat=flags.quantize_stat,
                        input_dir=flags.input_dir,
                        output_dir=flags.output_dir,
                        layers_validate_file=output_config[
                            YAMLKeyword.model_file_path],
                        pruning_type=pruning_type,
                        perf=flags.perf
                    )
                    if flags.validate:
                        model_file = model_config[YAMLKeyword.model_file_path]
                        if YAMLKeyword.validate_model_file_path in model_config:
                            model_file = model_config[YAMLKeyword.validate_model_file_path]
                        model_file_path, weight_file_path = get_model_files(
                            model_file,
                            BUILD_DOWNLOADS_DIR,
                            model_config[YAMLKeyword.weight_file_path]
                        )
                        validate_type = device_type
                        if model_config[YAMLKeyword.quantize] == 1:
                            validate_type = device_type + '_QUANTIZE'

                        command_helper.validate_model(
                            abi=target_abi,
                            device=self,
                            model_file_path=model_file_path,
                            weight_file_path=weight_file_path,
                            platform=model_config[YAMLKeyword.platform],
                            device_type=device_type,
                            input_nodes=subgraphs[0][
                                YAMLKeyword.input_tensors],
                            output_nodes=output_config[
                                YAMLKeyword.output_tensors],
                            input_shapes=subgraphs[0][
                                YAMLKeyword.input_shapes],
                            output_shapes=output_config[
                                YAMLKeyword.output_shapes],
                            input_data_formats=subgraphs[0][
                                YAMLKeyword.input_data_formats],
                            output_data_formats=subgraphs[0][
                                YAMLKeyword.output_data_formats],
                            model_output_dir=model_output_dir,
                            input_data_types=subgraphs[0][
                                YAMLKeyword.input_data_types],
                            validation_threshold=subgraphs[0][
                                YAMLKeyword.validation_threshold][
                                validate_type],
                            backend=subgraphs[0][YAMLKeyword.backend],
                            validation_outputs_data=subgraphs[0][
                                YAMLKeyword.validation_outputs_data],
                            log_file=log_file,
                        )
                    if flags.report and flags.round > 0:
                        tuned = is_tuned and device_type == DeviceType.GPU
                        self.report_run_statistics(
                            target_abi=target_abi,
                            model_name=model_name,
                            device_type=device_type,
                            output_dir=flags.report_dir,
                            tuned=tuned
                        )

    def report_run_statistics(self,
                              target_abi,
                              model_name,
                              device_type,
                              output_dir,
                              tuned):
        metrics = [0] * 3
        for line in self.stdout.split('\n'):
            line = line.strip()
            parts = line.split()
            if len(parts) == 5 and parts[0].startswith('time'):
                metrics[0] = str(float(parts[2]))
                metrics[1] = str(float(parts[3]))
                metrics[2] = str(float(parts[4]))
                break
        report_filename = output_dir + '/report.csv'
        if not os.path.exists(report_filename):
            with open(report_filename, 'w') as f:
                f.write('model_name,device_name,soc,abi,runtime,'
                        'init(ms),warmup(ms),run_avg(ms),tuned\n')

        data_str = '{model_name},{device_name},{soc},{abi},{device_type},' \
                   '{init},{warmup},{run_avg},{tuned}\n'.format(
                       model_name=model_name,
                       device_name=self.device_name,
                       soc=self.target_socs,
                       abi=target_abi,
                       device_type=device_type,
                       init=metrics[0],
                       warmup=metrics[1],
                       run_avg=metrics[2],
                       tuned=tuned)
        with open(report_filename, 'a') as f:
            f.write(data_str)

    def benchmark_model(self,
                        abi,
                        benchmark_binary_dir,
                        benchmark_binary_name,
                        vlog_level,
                        model_output_dir,
                        deepvan_model_dir,
                        input_nodes,
                        output_nodes,
                        input_shapes,
                        output_shapes,
                        input_data_formats,
                        output_data_formats,
                        max_num_runs,
                        max_seconds,
                        model_tag,
                        device_type,
                        opencl_binary_file,
                        opencl_parameter_file,
                        executor_dynamic_library_path,
                        omp_num_threads=-1,
                        cpu_affinity_policy=1,
                        gpu_perf_hint=3,
                        gpu_priority_hint=3,
                        input_file_name='model_input',
                        link_dynamic=False,
                        tuning_mode=False,
                        pruning_type=PruningType.DENSE,
                        perf=False):
        six.print_('* Benchmark for %s' % model_tag)
        deepvan_model_path = '%s/%s.pb' % (deepvan_model_dir, model_tag)

        model_data_file = ""
        other_data_file = ""
        if self.system == SystemType.host:
            model_data_file = "%s/%s.data" % (deepvan_model_dir, model_tag)
            if pruning_type == PruningType.PATTERN:
                other_data_file = "%s/%s.ptd" % (deepvan_model_dir, model_tag)
            elif pruning_type == PruningType.CSR:
                other_data_file = "%s/%s.csr" % (deepvan_model_dir, model_tag)
            elif pruning_type == PruningType.SLICE:
                # other_data_file = "%s/%s.slice" % (deepvan_model_dir, model_tag)
                pass
        else:
            model_data_file = "%s/%s.data" % (self.data_dir, model_tag)
            if pruning_type == PruningType.PATTERN:
                other_data_file = "%s/%s.ptd" % (self.data_dir, model_tag)
            elif pruning_type == PruningType.CSR:
                other_data_file = "%s/%s.csr" % (self.data_dir, model_tag)
            elif pruning_type == PruningType.SLICE:
                # other_data_file = "%s/%s.slice" % (self.data_dir, model_tag)
                pass
        if abi == ABIType.host:
            executor_dynamic_lib_dir_path = \
                os.path.dirname(executor_dynamic_library_path)
            p = subprocess.Popen(
                [
                    'env',
                    'LD_LIBRARY_PATH=%s' % executor_dynamic_lib_dir_path,
                    'DEEPVAN_CPP_MIN_VLOG_LEVEL=%s' % vlog_level,
                    '%s/%s' % (benchmark_binary_dir, benchmark_binary_name),
                    '--model_name=%s' % model_tag,
                    '--input_node=\"%s\"' % ','.join(input_nodes),
                    "--output_node=\"%s\"" % ",".join(output_nodes),
                    '--input_shape=%s' % ':'.join(input_shapes),
                    '--output_shape=%s' % ':'.join(output_shapes),
                    "--input_data_format=%s" % ",".join(input_data_formats),
                    "--output_data_format=%s" % ",".join(output_data_formats),
                    '--input_file=%s/%s' % (model_output_dir, input_file_name),
                    "--model_data_file=%s" % model_data_file,
                    "--other_data_file=%s" % other_data_file,
                    '--max_num_runs=%d' % max_num_runs,
                    '--max_seconds=%f' % max_seconds,
                    '--device=%s' % device_type,
                    '--omp_num_threads=%s' % omp_num_threads,
                    '--cpu_affinity_policy=%s' % cpu_affinity_policy,
                    '--gpu_perf_hint=%s' % gpu_perf_hint,
                    '--gpu_priority_hint=%s' % gpu_priority_hint,
                    '--model_file=%s' % deepvan_model_path
                ])
            p.wait()
        elif self.system in [SystemType.android, SystemType.arm_linux]:
            self.exec_command('mkdir -p %s' % self.data_dir)
            internal_storage_dir = self.create_internal_storage_dir()
            for input_name in input_nodes:
                formatted_name = formatted_file_name(input_file_name,
                                                     input_name)
                self.push('%s/%s' % (model_output_dir, formatted_name),
                          self.data_dir)
            model_data_path = "%s/%s.data" % (deepvan_model_dir, model_tag)
            model_data_exst = command_helper.compare_md5(
                model_data_path, "%s/%s.data" % (self.data_dir, model_tag), self.address)
            if not model_data_exst:
                self.push(model_data_path, self.data_dir)
            paten_data_path = "%s/%s.ptd" % (deepvan_model_dir, model_tag)
            if os.path.exists(paten_data_path):
                self.push(paten_data_path, self.data_dir)
            csr_data_path = "%s/%s.csr" % (deepvan_model_dir, model_tag)
            if os.path.exists(csr_data_path):
                self.push(csr_data_path, self.data_dir)
            slice_data_path = "%s/%s.slice" % (deepvan_model_dir, model_tag)
            if os.path.exists(slice_data_path):
                self.push(slice_data_path, self.data_dir)
            if device_type == common.DeviceType.GPU:
                if os.path.exists(opencl_binary_file):
                    self.push(opencl_binary_file, self.data_dir)
                if os.path.exists(opencl_parameter_file):
                    self.push(opencl_parameter_file, self.data_dir)
            deepvan_model_device_path = ''
            deepvan_model_device_path = '%s/%s.pb' % \
                                        (self.data_dir, model_tag)
            self.push(deepvan_model_path, deepvan_model_device_path)
            if link_dynamic:
                self.push(executor_dynamic_library_path, self.data_dir)
                if self.system == SystemType.android:
                    command_helper.push_depended_so_libs(
                        executor_dynamic_library_path, abi, self.data_dir,
                        self.address)
            self.rm('%s/%s' % (self.data_dir, benchmark_binary_name))
            self.push('%s/%s' % (benchmark_binary_dir, benchmark_binary_name),
                      self.data_dir)

            stdout_buff = []
            process_output = command_helper.make_output_processor(stdout_buff)
            target_with_params = [
                '%s/%s' % (self.data_dir, benchmark_binary_name),
                '--model_name=%s' % model_tag,
                '--input_node=\"%s\"' % ','.join(input_nodes),
                "--output_node=\"%s\"" % ",".join(output_nodes),
                '--input_shape=%s' % ':'.join(input_shapes),
                '--output_shape=%s' % ':'.join(output_shapes),
                "--input_data_format=%s" % ",".join(input_data_formats),
                "--output_data_format=%s" % ",".join(output_data_formats),
                '--input_file=%s/%s' % (self.data_dir, input_file_name),
                "--model_data_file=%s" % model_data_file,
                "--other_data_file=%s" % other_data_file,
                '--max_num_runs=%d' % max_num_runs,
                '--max_seconds=%f' % max_seconds,
                '--device=%s' % device_type,
                '--omp_num_threads=%s' % omp_num_threads,
                '--cpu_affinity_policy=%s' % cpu_affinity_policy,
                '--gpu_perf_hint=%s' % gpu_perf_hint,
                '--gpu_priority_hint=%s' % gpu_priority_hint,
                '--model_file=%s' % deepvan_model_device_path,
                '--opencl_binary_file=%s/%s' %
                (self.data_dir, os.path.basename(opencl_binary_file)),
                '--opencl_parameter_file=%s/%s' %
                (self.data_dir, os.path.basename(opencl_parameter_file)),
            ]
            perf_params = []
            if perf:
                if len(os.environ.get('SIMPLEPERF', "")) > 0:
                    self.push(os.environ['SIMPLEPERF'], self.data_dir)
                else:
                    self.push(command_helper.find_simpleperf_library(
                        abi), self.data_dir)
                simpleperf_cmd = '/data/local/tmp/simpleperf'
                perf_params = [
                    simpleperf_cmd,
                    'stat',
                    '--group',
                    'raw-l1-dcache,raw-l1-dcache-refill',
                    '--group',
                    'raw-l2-dcache,raw-l2-dcache-refill',
                    '--group',
                    'raw-l3-dcache,raw-l3-dcache-refill',
                    '--group',
                    'raw-l1-dtlb,raw-l1-dtlb-refill',
                    '--group',
                    'raw-l2-dtlb,raw-l2-dtlb-refill',
                ]
            cmd = [
                'LD_LIBRARY_PATH=%s' % self.data_dir,
                'DEEPVAN_CPP_MIN_VLOG_LEVEL=%s' % vlog_level,
                'DEEPVAN_RUN_PARAMETER_PATH=%s/deepvan_run.config' % self.data_dir,
                'DEEPVAN_INTERNAL_STORAGE_PATH=%s' % internal_storage_dir,
                'DEEPVAN_OPENCL_PROFILING=1',
            ] + perf_params + target_with_params

            time_stamp = str(time.time())
            cmd = ' '.join(cmd)
            cmd_file_name = '%s-%s-%s' % \
                            ('cmd_file', model_tag, time_stamp)

            cmd_file_path = '%s/%s' % (self.data_dir, cmd_file_name)
            tmp_cmd_file = '%s/%s' % ('/tmp', cmd_file_name)
            with open(tmp_cmd_file, 'w') as f:
                f.write(cmd)
            self.push(tmp_cmd_file, cmd_file_path)
            os.remove(tmp_cmd_file)
            # push debug file to target
            debug_cmd_file_name = "%s-%s-%s" % ('debug',
                                                model_tag,
                                                time_stamp)
            debug_cmd_file = "%s/%s" % (self.data_dir, debug_cmd_file_name)
            tmp_debug_cmd_file = "%s/%s" % ('/tmp', debug_cmd_file_name)
            debug_cmd = "/data/local/tmp/gdbserver :10010 " + \
                ' '.join(target_with_params)
            with open(tmp_debug_cmd_file, 'w') as file:
                file.write(debug_cmd)
            self.push(tmp_debug_cmd_file, debug_cmd_file)
            os.remove(tmp_debug_cmd_file)

            if self.system == SystemType.android:
                self.exec_command('sh {}'.format(cmd_file_path),
                                  _tty_in=True,
                                  _out=process_output,
                                  _err_to_out=True)
            elif self.system == SystemType.arm_linux:
                sh.ssh('%s@%s' % (self.username, self.address),
                       'sh', cmd_file_path, _fg=True)

    def bm_specific_target(self, flags, configs, target_abi, tuning_mode=False, pruning_type=PruningType.DENSE):
        library_name = configs[YAMLKeyword.library_name]
        opencl_output_bin_path = ''
        opencl_parameter_path = ''
        link_dynamic = flags.deepvan_lib_type == DEEPVANLibType.dynamic

        if link_dynamic:
            bm_model_binary_name = BM_MODEL_DYNAMIC_NAME
        else:
            bm_model_binary_name = BM_MODEL_STATIC_NAME
        build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
        if (configs[YAMLKeyword.target_socs] or flags.target_socs)\
                and target_abi != ABIType.host:
            # TODO @vgod
            opencl_output_bin_path = ''
            opencl_parameter_path = ''

        for model_name in configs[YAMLKeyword.models]:
            check_model_converted(library_name, model_name, target_abi)
            DLogger.header(
                StringFormatter.block(
                    'Benchmark model %s on %s' % (model_name,
                                                  self.device_name)))
            model_config = configs[YAMLKeyword.models][model_name]
            model_runtime = model_config[YAMLKeyword.runtime]
            subgraphs = model_config[YAMLKeyword.subgraphs]

            model_output_base_dir, model_output_dir, deepvan_model_dir = \
                get_build_model_dirs(library_name, model_name,
                                     target_abi, self,
                                     model_config[YAMLKeyword.model_file_path])
            if os.path.exists(model_output_dir):
                sh.rm('-rf', model_output_dir)
            os.makedirs(model_output_dir)

            if target_abi != ABIType.host:
                self.clear_data_dir()
            command_helper.gen_random_input(
                model_output_dir,
                subgraphs[0][YAMLKeyword.input_tensors],
                subgraphs[0][YAMLKeyword.input_shapes],
                subgraphs[0][YAMLKeyword.validation_inputs_data],
                input_ranges=subgraphs[0][YAMLKeyword.input_ranges],
                input_data_types=subgraphs[0][YAMLKeyword.input_data_types]
            )
            runtime_list = []
            if target_abi == ABIType.host:
                runtime_list.append(RuntimeType.cpu)
            elif model_runtime == RuntimeType.cpu_gpu:
                runtime_list.extend([RuntimeType.cpu, RuntimeType.gpu])
            else:
                runtime_list.append(model_runtime)
            for runtime in runtime_list:
                device_type = parse_device_type(runtime)
                if not subgraphs[0][YAMLKeyword.check_tensors]:
                    output_nodes = subgraphs[0][YAMLKeyword.output_tensors]
                    output_shapes = subgraphs[0][YAMLKeyword.output_shapes]
                else:
                    output_nodes = subgraphs[0][YAMLKeyword.check_tensors]
                    output_shapes = subgraphs[0][YAMLKeyword.check_shapes]
                result = self.benchmark_model(
                    abi=target_abi,
                    benchmark_binary_dir=build_tmp_binary_dir,
                    benchmark_binary_name=bm_model_binary_name,
                    vlog_level=0,
                    model_output_dir=model_output_dir,
                    input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
                    output_nodes=output_nodes,
                    input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
                    output_shapes=output_shapes,
                    input_data_formats=subgraphs[0][
                        YAMLKeyword.input_data_formats],
                    output_data_formats=subgraphs[0][
                        YAMLKeyword.output_data_formats],
                    max_num_runs=flags.max_num_runs,
                    max_seconds=flags.max_seconds,
                    deepvan_model_dir=deepvan_model_dir,
                    model_tag=model_name,
                    device_type=device_type,
                    omp_num_threads=flags.omp_num_threads,
                    cpu_affinity_policy=flags.cpu_affinity_policy,
                    gpu_perf_hint=flags.gpu_perf_hint,
                    gpu_priority_hint=flags.gpu_priority_hint,
                    opencl_binary_file=opencl_output_bin_path,
                    opencl_parameter_file=opencl_parameter_path,
                    executor_dynamic_library_path=LIBDEEPVAN_DYNAMIC_PATH,
                    link_dynamic=link_dynamic,
                    tuning_mode=tuning_mode,
                    pruning_type=pruning_type,
                    perf=flags.perf)
                if tuning_mode and result != None:
                    return result

    def run(self,
            abi,
            host_bin_path,
            bin_name,
            args='',
            opencl_profiling=True,
            vlog_level=0,
            out_of_range_check=True,
            address_sanitizer=False,
            simpleperf=False):
        host_bin_full_path = '%s/%s' % (host_bin_path, bin_name)
        device_bin_full_path = '%s/%s' % (self.data_dir, bin_name)
        print(
            '================================================================'
        )
        print('Trying to lock device %s' % self.address)
        with self.lock():
            print('Run on device: %s, %s, %s' %
                  (self.address, self.target_socs, self.device_name))
            self.rm(self.data_dir)
            self.exec_command('mkdir -p %s' % self.data_dir)
            self.push(host_bin_full_path, device_bin_full_path)
            ld_preload = ''
            if address_sanitizer:
                self.push(command_helper.find_asan_rt_library(abi),
                          self.data_dir)
                ld_preload = 'LD_PRELOAD=%s/%s' % \
                             (self.data_dir,
                              command_helper.asan_rt_library_names(abi))
            opencl_profiling = 1 if opencl_profiling else 0
            out_of_range_check = 1 if out_of_range_check else 0
            print('Run %s' % device_bin_full_path)
            stdout_buf = []
            process_output = command_helper.make_output_processor(stdout_buf)

            internal_storage_dir = self.create_internal_storage_dir()

            if simpleperf and self.system == SystemType.android:
                self.push(command_helper.find_simpleperf_library(abi),
                          self.data_dir)
                simpleperf_cmd = '/data/local/tmp/simpleperf'
                exec_cmd = [
                    ld_preload,
                    'OUT_OF_RANGE_CHECK=%s' % out_of_range_check,
                    'DEEPVAN_OPENCL_PROFILING=%d' % opencl_profiling,
                    'DEEPVAN_INTERNAL_STORAGE_PATH=%s' % internal_storage_dir,
                    'DEEPVAN_CPP_MIN_VLOG_LEVEL=%d' % vlog_level,
                    simpleperf_cmd,
                    'stat',
                    '--group',
                    'raw-l1-dcache,raw-l1-dcache-refill',
                    '--group',
                    'raw-l2-dcache,raw-l2-dcache-refill',
                    '--group',
                    'raw-l1-dtlb,raw-l1-dtlb-refill',
                    '--group',
                    'raw-l2-dtlb,raw-l2-dtlb-refill',
                    device_bin_full_path,
                    args,
                ]
            else:
                exec_cmd = [
                    ld_preload,
                    'OUT_OF_RANGE_CHECK=%d' % out_of_range_check,
                    'DEEPVAN_OPENCL_PROFILING=%d' % opencl_profiling,
                    'DEEPVAN_INTERNAL_STORAGE_PATH=%s' % internal_storage_dir,
                    'DEEPVAN_CPP_MIN_VLOG_LEVEL=%d' % vlog_level,
                    device_bin_full_path,
                    args
                ]
            exec_cmd = ' '.join(exec_cmd)
            # save debug file
            target_with_params = [
                device_bin_full_path,
                args
            ]
            debug_cmd_file_name = "%s-%s" % ('debug', bin_name)
            debug_cmd_file_path = "%s/%s" % (self.data_dir,
                                             debug_cmd_file_name)
            tmp_debug_cmd_file = "%s/%s" % ('/tmp', debug_cmd_file_name)
            debug_cmd = "/data/local/tmp/gdbserver :10010 " + \
                ' '.join(target_with_params)
            with open(tmp_debug_cmd_file, 'w') as file:
                file.write(debug_cmd)
            self.push(tmp_debug_cmd_file, debug_cmd_file_path)
            os.remove(tmp_debug_cmd_file)
            self.exec_command(exec_cmd, _tty_in=True,
                              _out=process_output, _err_to_out=True)
            return ''.join(stdout_buf)


class DeviceManager:
    @classmethod
    def list_adb_device(cls):
        adb_list = sh.adb('devices').stdout.decode('utf-8'). \
            strip().split('\n')[1:]
        adb_list = [tuple(pair.split('\t')) for pair in adb_list]
        devices = []
        for adb in adb_list:
            if adb[1].startswith("no permissions") or adb[1].startswith('unauthorized'):
                continue
            prop = command_helper.adb_getprop_by_serialno(adb[0])
            android = {
                YAMLKeyword.device_name:
                    prop['ro.product.model'].replace(' ', ''),
                YAMLKeyword.target_abis:
                    prop['ro.product.cpu.abilist'].split(','),
                YAMLKeyword.target_socs: prop['ro.board.platform'],
                YAMLKeyword.system: SystemType.android,
                YAMLKeyword.address: adb[0],
                YAMLKeyword.username: '',
            }
            if android not in devices:
                devices.append(android)
        return devices

    @classmethod
    def list_ssh_device(cls, yml):
        with open(yml) as f:
            devices = yaml.load(f.read())
        devices = devices['devices']
        device_list = []
        for name, dev in six.iteritems(devices):
            dev[YAMLKeyword.device_name] = \
                dev[YAMLKeyword.models].replace(' ', '')
            dev[YAMLKeyword.system] = SystemType.arm_linux
            device_list.append(dev)
        return device_list

    @classmethod
    def list_devices(cls, yml):
        devices_list = []
        devices_list.extend(cls.list_adb_device())
        if not yml:
            if os.path.exists('devices.yml'):
                devices_list.extend(cls.list_ssh_device('devices.yml'))
        else:
            if os.path.exists(yml):
                devices_list.extend(cls.list_ssh_device(yml))
            else:
                DLogger.error(ModuleName.RUN,
                              'no ARM linux device config file found')
        host = {
            YAMLKeyword.device_name: SystemType.host,
            YAMLKeyword.target_abis: [ABIType.host],
            YAMLKeyword.target_socs: '',
            YAMLKeyword.system: SystemType.host,
            YAMLKeyword.address: None,

        }
        devices_list.append(host)
        return devices_list


if __name__ == '__main__':
    pass
