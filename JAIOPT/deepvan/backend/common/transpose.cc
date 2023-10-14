#include "deepvan/backend/common/transpose.h"

#if defined(NEON_SUPPORT)
#include <arm_neon.h>
#endif

namespace deepvan {
namespace transpose {

void TransposeNHWCToNCHWC3(const float *input,
                           float *output,
                           const index_t height,
                           const index_t width) {
  index_t image_size = height * width;

#pragma omp parallel for
  for (index_t h = 0; h < height; ++h) {
    index_t in_offset = h * width * 3;
    index_t out_offset = h * width;

#if defined(NEON_SUPPORT)
    index_t w;
    for (w = 0; w + 3 < width; w += 4) {
      float32x4x3_t vi = vld3q_f32(input + in_offset);
      vst1q_f32(output + out_offset, vi.val[0]);
      vst1q_f32(output + out_offset + image_size, vi.val[1]);
      vst1q_f32(output + out_offset + image_size * 2, vi.val[2]);

      in_offset += 12;
      out_offset += 4;
    }
    for (; w < width; ++w) {
      for (index_t c = 0; c < 3; ++c) {
        output[h * width + image_size * c + w] =
          input[h * width * 3 + w * 3 + c];
      }
    }
#else
    for (index_t w = 0; w < width; ++w) {
      for (index_t c = 0; c < 3; ++c) {
        output[out_offset + c * image_size + w] = input[in_offset + w * 3 + c];
      }
    }
#endif
  }
}

void TransposeNCHWToNHWCC2(const float *input,
                           float *output,
                           const index_t height,
                           const index_t width) {
  index_t image_size = height * width;
#pragma omp parallel for
  for (index_t h = 0; h < height; ++h) {
    index_t in_offset = h * width;
    index_t out_offset = h * width * 2;

#if defined(NEON_SUPPORT)
    index_t w;
    for (w = 0; w + 3 < width; w += 4) {
      float32x4_t vi0 = vld1q_f32(input + in_offset);
      float32x4_t vi1 = vld1q_f32(input + in_offset + image_size);
      float32x4x2_t vi = {vi0, vi1};
      vst2q_f32(output + out_offset, vi);
      in_offset += 4;
      out_offset += 8;
    }
    for (; w < width; ++w) {
      for (index_t c = 0; c < 2; ++c) {
        output[h * width * 2 + w * 2 + c] =
          input[h * width + image_size * c + w];
      }
    }
#else
    for (index_t w = 0; w < width; ++w) {
      for (index_t c = 0; c < 2; ++c) {
        output[out_offset + w * 2 + c] = input[in_offset + c * image_size + w];
      }
    }
#endif
  }
}

void TransposeNHWCToNCHWC3(const int *input,
                           int *output,
                           const index_t height,
                           const index_t width) {
  index_t image_size = height * width;

#pragma omp parallel for
  for (index_t h = 0; h < height; ++h) {
    index_t in_offset = h * width * 3;
    index_t out_offset = h * width;

    for (index_t w = 0; w < width; ++w) {
      for (index_t c = 0; c < 3; ++c) {
        output[out_offset + c * image_size + w] = input[in_offset + w * 3 + c];
      }
    }
  }
}

void TransposeNCHWToNHWCC2(const int *input,
                           int *output,
                           const index_t height,
                           const index_t width) {
  index_t image_size = height * width;
#pragma omp parallel for
  for (index_t h = 0; h < height; ++h) {
    index_t in_offset = h * width;
    index_t out_offset = h * width * 2;

    for (index_t w = 0; w < width; ++w) {
      for (index_t c = 0; c < 2; ++c) {
        output[out_offset + w * 2 + c] = input[in_offset + c * image_size + w];
      }
    }
  }
}
}  // namespace transpose
}  // namespace deepvan
