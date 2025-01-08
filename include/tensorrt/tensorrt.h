#ifndef __OPENRM_TENSORRT_TENSORRT_H__
#define __OPENRM_TENSORRT_TENSORRT_H__

//#include <NvInfer.h>
//#include <NvInferRuntime.h>
//#include <NvOnnxParser.h>
#include <string>
#include "structure/stamp.hpp"      //定义了如 YoloRect 这样的结构体（用于非极大值抑制 NMS 函数）
//#include "tensorrt/logging.h"
#include <openvino/openvino.hpp>

namespace rm {

// bool initTrtOnnx(
//     const std::string& onnx_file,
//     const std::string& engine_file,
//     //nvinfer1::IExecutionContext** context,
//     unsigned int batch_size = 1U
// );

// bool initTrtEngine(
//     const std::string& engine_file
//     //nvinfer1::IExecutionContext** context
// );

bool initOpenVINO(const std::string& model_path, ov::Core& core, ov::CompiledModel& compilemodel);


// bool initCudaStream(
//     //cudaStream_t* stream
// );

// void detectEnqueue(
//     float* input_device_buffer,
//     float* output_device_buffer
//     //nvinfer1::IExecutionContext** context,
//     //cudaStream_t* stream
// );

// 执行推理
void detectInfer(const float* input_host_buffer, float* output_host_buffer, ov::CompiledModel& compiled_model);

// void detectOutput(
//     float* output_host_buffer,
//     const float* output_device_buffer,
//     //cudaStream_t* stream,
//     size_t output_struct_size,
//     int bboxes_num,
//     int batch_size = 1
// ); 

void detectOutput(
    float* output_host_buffer,
    const ov::Tensor& output_tensor,
    size_t output_struct_size,
    int bboxes_num,
    int batch_size = 1
);

// void detectOutputClassify(
//     float* output_host_buffer,
//     const float* output_device_buffer,
//     //cudaStream_t* stream,
//     int class_num
// ); 

void detectOutputClassify(
    float* output_host_buffer,
    const ov::Tensor& output_tensor,
    int class_num
);


// void mallocYoloCameraBuffer(
//     uint8_t** rgb_host_buffer,
//     uint8_t** rgb_device_buffer,
//     int rgb_width,
//     int rgb_height,
//     int batch_size = 1,
//     int channels = 3
// );


// void mallocYoloDetectBuffer(
//     float** input_device_buffer,
//     float** output_device_buffer,
//     float** output_host_buffer,
//     int input_width,
//     int input_height,
//     size_t output_struct_size,
//     int bboxes_num,
//     int batch_size = 1,
//     int channels = 3
// );

// void mallocClassifyBuffer(
//     float** input_host_buffer,
//     float** input_device_buffer,
//     float** output_device_buffer,
//     float** output_host_buffer,
//     int input_width,
//     int input_height,
//     int class_num,
//     int channels = 3
// );

// 通用内存分配函数
void mallocHostBuffer(float** input_host_buffer, float** output_host_buffer, size_t input_size, size_t output_size);

// 摄像头输入任务的内存分配函数
void mallocCameraBuffer(float** input_host_buffer, float** output_host_buffer, size_t input_size, size_t output_size);

// YOLO 检测任务的内存分配函数
void mallocDetectBuffer(
    float** input_host_buffer,
    float** output_host_buffer,
    int input_width,
    int input_height,
    size_t output_struct_size,
    int bboxes_num,
    int batch_size,
    int channels
); 

// 分类任务的内存分配函数
void mallocClassifyBuffer(
    float** input_host_buffer,
    float** output_host_buffer,
    int input_width,
    int input_height,
    int class_num,
    int channels = 3
);


// void freeYoloCameraBuffer(
//     uint8_t* rgb_host_buffer,
//     uint8_t* rgb_device_buffer
// );


// void freeYoloDetectBuffer(
//     float* input_device_buffer,
//     float* output_device_buffer,
//     float* output_host_buffer
// );

// void freeClassifyBuffer(
//     float* input_host_buffer,
//     float* input_device_buffer,
//     float* output_device_buffer,
//     float* output_host_buffer
// );

void freeHostBuffer(void* host_buffer);

// void memcpyYoloCameraBuffer(
//     uint8_t* rgb_mat_data,
//     uint8_t* rgb_host_buffer,
//     uint8_t* rgb_device_buffer,
//     int rgb_width,
//     int rgb_height,
//     int channels = 3
// );

// void memcpyClassifyBuffer(
//     uint8_t* mat_data,
//     float* input_host_buffer,
//     float* input_device_buffer,
//     int input_width,
//     int input_height,
//     int channels = 3
// );

std::vector<YoloRect> yoloArmorNMS_V5C36(
    float* output_host_buffer,
    int output_bboxes_num,
    int armor_classes_num,
    float confidence_threshold,
    float nms_threshold,
    int input_width,
    int input_height,
    int infer_width,
    int infer_height
);

std::vector<YoloRect> yoloArmorNMS_V5(
    float* output_host_buffer,
    int output_bboxes_num,
    int armor_classes_num,
    float confidence_threshold,
    float nms_threshold,
    int input_width,
    int input_height,
    int infer_width,
    int infer_height
);

std::vector<YoloRect> yoloArmorNMS_FP(
    float* output_host_buffer,
    int output_bboxes_num,
    int classes_num,
    float confidence_threshold,
    float nms_threshold,
    int input_width,
    int input_height,
    int infer_width,
    int infer_height
);

std::vector<YoloRect> yoloArmorNMS_FPX(
    float* output_host_buffer,
    int output_bboxes_num,
    int classes_num,
    float confidence_threshold,
    float nms_threshold,
    int input_width,
    int input_height,
    int infer_width,
    int infer_height
);

}

#endif
