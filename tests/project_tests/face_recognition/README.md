# Face Recognition

## 简介

人脸识别任务主要使用检测器+识别器构成，EasyDeploy中把它们整合成了FaceRecognition类，共有PC和Board两个版本。

在这个demo中，我们以SCRFD作为检测器，MobileFacenet-AdaFace作为识别器。

## 使用教程

在板子上部署前，请先运行PC端代码，生成rknn文件

### python

**PC**
****

```text
cd python
python face_recognition_pc.py
```

**Board**
****

```text
cd python
python face_recognition_board.py
```