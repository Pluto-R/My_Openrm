#include "resize.h"

void rm::resize(
    uint8_t* src,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height
    // void* _cuda_stream
) {
    // void* _cuda_stream
    // 生成仿射变换矩阵
    generate_affine_matrix(src_width, src_height, dst_width, dst_height);

    //114 是一个常用的填充值，因为它是一个中等灰度值，不会对模型的训练和推理产生太大影响。
    warpaffine_opencv(src,src_width,src_height,dst,dst_width,dst_height,114,infer_to_input);

    // // 计算线程块和线程数量
    // int jobs = dst_width * dst_height;
    // int threads = 256;
    // int blocks = ceil(jobs / (float)threads);

    // // 启动核函数
    // warpaffine_kernel<<<blocks, threads, 0, cuda_stream>>>(
    //    src,
    //    src_width * 3,
    //    src_width,
    //    src_height,
    //    dst,
    //    dst_width,
    //    dst_height,
    //    114,
    //    infer_to_input,
    //    jobs
    // );

    // // 等待核函数执行完成
    // cudaStreamSynchronize(cuda_stream);
}
