#include <common.h>
   
__kernel void activation4d_tensor(OUT_OF_RANGE_PARAMS
                         GLOBAL_WORK_GROUP_SIZE_DIM3
                         __read_only image2d_t input,
#ifdef USE_PRELU
                         __read_only image2d_t alpha,
#endif
                         __private const float relux_max_limit,
                         __private const float leakyrelu_coefficient,
                         __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1); 
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (ch_blk >= global_size_dim0 || w >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

  const int pos = mad24(ch_blk, width, w);
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb));
#ifdef USE_PRELU
  DATA_TYPE4 prelu_alpha = READ_IMAGET(alpha, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 out = do_activation(in, prelu_alpha, relux_max_limit, leakyrelu_coefficient);
#else
  DATA_TYPE4 out = do_activation(in, relux_max_limit, leakyrelu_coefficient);
#endif

  WRITE_IMAGET(output, (int2)(pos, hb), out);
}



__kernel void activation2d_tensor(OUT_OF_RANGE_PARAMS
                                  GLOBAL_WORK_GROUP_SIZE_DIM2
                                  __read_only image2d_t input,
#ifdef USE_PRELU
                                  __read_only image2d_t alpha,
#endif
                                  __private const float relux_max_limit,
                                  __private const float leakyrelu_coefficient,
                                  __write_only image2d_t output) {
  const int wb = get_global_id(0);
  const int hb = get_global_id(1);
  
  const int ch_blk = wb;

  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(wb, hb));
#ifdef USE_PRELU
  DATA_TYPE4 prelu_alpha = READ_IMAGET(alpha, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 out = do_activation(in, prelu_alpha, relux_max_limit, leakyrelu_coefficient);
#else
  DATA_TYPE4 out = do_activation(in, relux_max_limit, leakyrelu_coefficient);
#endif

  WRITE_IMAGET(output, (int2)(wb, hb), out);
}

__kernel void activation_bert(OUT_OF_RANGE_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM2
                              __read_only image2d_t input,
#ifdef USE_PRELU
                              __read_only image2d_t alpha,
#endif
                              __private const float relux_max_limit,
                              __private const float leakyrelu_coefficient,
                              __write_only image2d_t output) {
  const int wb = get_global_id(0);
  const int hb = get_global_id(1);
  const int ch_blk = wb;

  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(wb, hb));
#ifdef USE_PRELU
  DATA_TYPE4 prelu_alpha = READ_IMAGET(alpha, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 out = do_activation(in, prelu_alpha, relux_max_limit, leakyrelu_coefficient);
#else
  DATA_TYPE4 out = do_activation(in, relux_max_limit, leakyrelu_coefficient);
#endif

  WRITE_IMAGET(output, (int2)(wb, hb), out);
}
