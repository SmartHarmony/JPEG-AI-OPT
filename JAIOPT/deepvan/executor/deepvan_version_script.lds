deepvan {
  global:
    *GPUContextBuilder*;
    *ModelExecutionConfig*;
    *TensorWrapper*;
    *ModelExecutor*;
    *CreateModelExecutorFromProto*;
    *GetBigLittleCoreIDs*;
    *GetCapability*;

    # api for static library of models
    *deepvan*compat**;
    *deepvan*VanState*;
    *deepvan*NetProto*;
    *deepvan*MemoryType*;
    *deepvan*DataType*;
    *deepvan*InputOutputInfo*;
    *deepvan*OutputShape*;
    *deepvan*OperatorProto*;
    *deepvan*TensorProto*;
    *deepvan*Argument*;
    *deepvan*Split*;
    *deepvan*MemoryBlock*;
    *google*protobuf*;

  local:
    *;
};
