#ifndef __OPENRM_RESIZE_CUH__
#define __OPENRM_RESIZE_CUH__

#include "cudatools.h"

struct AffineMatrix {
    float value[6];
};

extern AffineMatrix input_to_infer;
extern AffineMatrix infer_to_input;

void generate_affine_matrix(int src_width, int src_height, int dst_width, int dst_height);

// void warpaffine_kernel(
//     uint8_t* src,
//     int src_line_size,
//     int src_width,
//     int src_height,
//     float* dst,
//     int dst_width,
//     int dst_height,
//     uint8_t const_value_st,
//     AffineMatrix d2s,
//     int edge
// );

void warpaffine_opencv(
    uint8_t* src_data,          // 输入图像数据（BGR 格式）
    int src_width,              
    int src_height,             
    float* dst_data,            // 输出图像数据（RGB 格式，归一化到 [0, 1]）
    int dst_width,              
    int dst_height,             
    uint8_t const_value_st,     // 超出边界的填充值
    AffineMatrix d2s            // 仿射变换矩阵（6 个 float 值）
);

#endif
