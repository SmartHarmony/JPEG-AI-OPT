output_file_path=""
object_files=""
workspace=`mktemp -d ./tmpd.XXXXXX`
ar_command=$1

while read script_line; do
    command=""
    lib_path=""
    eval $(echo ${script_line} | awk -F" " \
            '{printf("command=%s\nlib_path=%s", $1, $2);}')
    upper_command=`echo ${command} | tr 'a-z' 'A-Z'`
    if [[ ${upper_command} == "CREATE" ]]; then
        output_file_path=${lib_path}
    elif [[ ${upper_command} == "ADDLIB" ]]; then
        lib_name=$(basename ${lib_path})
        lib_dir=${workspace}"/"${lib_name}
        mkdir ${lib_dir}
        cp ${lib_path} ${lib_dir}
        cur_path=`pwd`
        cd ${lib_dir}
        ${cur_path}"/"${ar_command} -x ${lib_name}
        object_files=${object_files}" "${lib_dir}"/*.o"
        cd ${cur_path}
    elif [[ ${upper_command} == "SAVE" ]]; then
        ${ar_command} -rcsu $output_file_path ${object_files}
    elif [[ ${upper_command} == "END" ]]; then
        echo "========== ar_merge_on_darwin end =========="
    else
        echo "error: Get an invalid input line: "$script_line
    fi
done
