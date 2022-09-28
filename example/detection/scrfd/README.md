# SCRFD
SCRFD是一种高效的轻量级高精度人脸检测方法，被ICLR-2022接受。

## 下载预训练ONNX模型

以下模型来自[FastDeploy](https://github.com/PaddlePaddle/FastDeploy/blob/develop/examples/vision/facedet/scrfd/README.md?plain=1)

| 模型                                                                                                   | 大小    | 精度     |
|:-----------------------------------------------------------------------------------------------------|:------|:-------|
| [SCRFD-500M-kps-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape160x160.onnx)  | 2.5MB | -      |
| [SCRFD-500M-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_shape160x160.onnx)            | 2.2MB | -      |
| [SCRFD-500M-kps-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape320x320.onnx)  | 2.5MB | -      |
| [SCRFD-500M-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_shape320x320.onnx)            | 2.2MB | -      |
| [SCRFD-500M-kps-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_bnkps_shape640x640.onnx)  | 2.5MB | 90.97% |
| [SCRFD-500M-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_500m_shape640x640.onnx)            | 2.2MB | 90.57% |
| [SCRFD-1G-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_1g_shape160x160.onnx )               | 2.5MB | -      |
| [SCRFD-1G-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_1g_shape320x320.onnx)                | 2.5MB | -      |
| [SCRFD-1G-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_1g_shape640x640.onnx)                | 2.5MB | 92.38% |
| [SCRFD-2.5G-kps-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape160x160.onnx)  | 3.2MB | -      |
| [SCRFD-2.5G-160](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_shape160x160.onnx)            | 2.6MB | -      |
| [SCRFD-2.5G-kps-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape320x320.onnx)  | 3.2MB | -      |
| [SCRFD-2.5G-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_shape320x320.onnx)            | 2.6MB | -      |
| [SCRFD-2.5G-kps-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape640x640.onnx)  | 3.2MB | 93.8%  |
| [SCRFD-2.5G-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_shape640x640.onnx)            | 2.6MB | 93.78% |
| [SCRFD-10G-kps-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_bnkps_shape320x320.onnx)    | 17MB  | -      |
| [SCRFD-10G-320](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_shape320x320.onnx)              | 15MB  | -      |
| [SCRFD-10G-kps-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_bnkps_shape640x640.onnx)    | 17MB  | 95.4%  |
| [SCRFD-10G-640](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_shape640x640.onnx)              | 15MB  | 95.16% |
| [SCRFD-10G-kps-1280](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_bnkps_shape1280x1280.onnx) | 17MB  | -      |
| [SCRFD-10G-1280](https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_10g_shape1280x1280.onnx)           | 15MB  | -      |

## 详细部署文档

- [Python部署](python)

## 参考仓库🙏🙏🙏

原仓库地址: [insightface](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

代码推理参考: [FastDeploy](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/facedet/scrfd)