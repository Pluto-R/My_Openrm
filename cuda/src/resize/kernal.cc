#include "resize.h"
#include "cudatools.h"
#include <opencv2/opencv.hpp>
#include <vector>

void warpaffine_opencv(
    uint8_t* src_data,          // 输入图像数据（BGR 格式）
    int src_width,              
    int src_height,             
    float* dst_data,            // 输出图像数据（RGB 格式，归一化到 [0, 1]）
    int dst_width,              
    int dst_height,             
    uint8_t const_value_st,     // 超出边界的填充值
    AffineMatrix d2s            // 仿射变换矩阵（6 个 float 值）
) {
    // 将输入数据转换为 OpenCV Mat
    cv::Mat src(src_height, src_width, CV_8UC3, src_data);

    // 定义仿射变换矩阵
    cv::Mat affine_matrix(2,3,CV_32F,d2s.value);

    // 定义输出图像尺寸
    cv::Size dst_size(dst_width, dst_height);

    // 应用仿射变换(使用双线性插值）
    cv::Mat dst;
    cv::warpAffine(src, dst, affine_matrix, dst_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(const_value_st, const_value_st, const_value_st));

    // 将 BGR 转换为 RGB
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    // 归一化到 [0, 1] 范围
    dst.convertTo(dst, CV_32F, 1.0 / 255.0);

    // 将输出数据按通道分离存储（rrrgggbbb 格式）
    int area = dst_width * dst_height;
    std::vector<cv::Mat> channels(3);
    cv::split(dst, channels);

    for (int i = 0; i < 3; i++) {
        std::memcpy(dst_data + i * area, channels[i].data, area * sizeof(float));
    }
}

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
//     int edge        //目标图像的总像素数
// ) {
    //int position = blockDim.x * blockIdx.x + threadIdx.x;
    //if (position >= edge)
    //    return;

    //float m_x1 = d2s.value[0];
    //float m_y1 = d2s.value[1];
    //float m_z1 = d2s.value[2];
    //float m_x2 = d2s.value[3];
    //float m_y2 = d2s.value[4];
    //float m_z2 = d2s.value[5];

    //int dx = position % dst_width;
    //int dy = position / dst_width;
    //float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    //float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    //float c0, c1, c2;

    //if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    //    // out of range
    //    c0 = const_value_st;
    //    c1 = const_value_st;
    //    c2 = const_value_st;
    //} else {
    //    int y_low = floorf(src_y);
    //    int x_low = floorf(src_x);
    //    int y_high = y_low + 1;
    //    int x_high = x_low + 1;

    //    uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
    //    float ly = src_y - y_low;
    //    float lx = src_x - x_low;
    //    float hy = 1 - ly;
    //    float hx = 1 - lx;
    //    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    //    uint8_t* v1 = const_value;
    //    uint8_t* v2 = const_value;
    //    uint8_t* v3 = const_value;
    //    uint8_t* v4 = const_value;

    //    if (y_low >= 0) {
    //        if (x_low >= 0)
    //            v1 = src + y_low * src_line_size + x_low * 3;

    //        if (x_high < src_width)
    //            v2 = src + y_low * src_line_size + x_high * 3;
    //    }

    //    if (y_high < src_height) {
    //        if (x_low >= 0)
    //            v3 = src + y_high * src_line_size + x_low * 3;

    //        if (x_high < src_width)
    //            v4 = src + y_high * src_line_size + x_high * 3;
    //    }

    //    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    //    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    //    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    //}

    //// bgr to rgb
    //float t = c2;
    //c2 = c0;
    //c0 = t;

    //// normalization
    //// printf("c0: %f, c1: %f, c2: %f\n", c0, c1, c2);
    //c0 = c0 / 255.0f;
    //c1 = c1 / 255.0f;
    //c2 = c2 / 255.0f;

    //// rgbrgbrgb to rrrgggbbb
    //int area = dst_width * dst_height;
    //float* pdst_c0 = dst + dy * dst_width + dx;
    //float* pdst_c1 = pdst_c0 + area;
    //float* pdst_c2 = pdst_c1 + area;
    //*pdst_c0 = c0;
    //*pdst_c1 = c1;
    //*pdst_c2 = c2;
//}
