# ArcFace
该模型简介出自[FastDeploy](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/faceid/insightface)

## 训练与导出ONNX模型

训练与导出模型请参考[ArcFace原仓库](https://github.com/deepinsight/insightface/commit/babb9a5)

## 下载预训练ONNX模型
FastDeploy中提供了转换好的ONNX模型，详情参考FastDeploy仓库

| 模型                                                                                     | 大小    | 精度 (AgeDB_30) |
|:---------------------------------------------------------------------------------------|:------|:--------------|
| [ArcFace-r18](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r18.onnx)      | 92MB  | 97.7          |
| [ArcFace-r34](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r34.onnx)      | 131MB | 98.1          |
| [ArcFace-r50](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r50.onnx)      | 167MB | -             |
| [ArcFace-r100](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx)    | 249MB | 98.4          |
| [ArcFace-r100_lr0.1](https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_r100_lr01.onnx) | 249MB | 98.4          |


## 详细部署文档

- [Python部署](python)