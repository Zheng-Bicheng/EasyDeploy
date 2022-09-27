# AdaFace


## 简介
一直以来，低质量图像的人脸识别都具有挑战性，因为低质量图像的人脸属性是模糊和退化的。将这样的图片输入模型时，将不能很好的实现分类。
而在人脸识别任务中，我们经常会利用opencv的仿射变换来矫正人脸数据，这时数据会出现低质量退化的现象。如何解决低质量图片的分类问题成为了模型落地时的痛点问题。

在AdaFace这项工作中，作者在损失函数中引入了另一个因素，即图像质量。作者认为，强调错误分类样本的策略应根据其图像质量进行调整。
具体来说，简单或困难样本的相对重要性应该基于样本的图像质量来给定。据此作者提出了一种新的损失函数来通过图像质量强调不同的困难样本的重要性。

由上，AdaFace缓解了低质量图片在输入网络后输出结果精度变低的情况，更加适合在人脸识别任务落地中使用。

## 模型精度
EasyDeploy中使用的模型是我们使用MobileFaceNet + AdaFace训练出来的，网络仅4.9M。其精确度如下:

| Arch          | Dataset | Method  | LFW     | CFPFP   | CPLFW   | CALFW   | AGEDB   | AVG     |
|---------------|---------|---------|---------|---------|---------|---------|---------|---------|
| MobileFacenet | MS1MV2  | AdaFace | 0.99417 | 0.92229 | 0.88250 | 0.95200 | 0.95767 | 0.94172 |

## 训练自己的数据
如果你需要训练自己的数据集，请参考[AdaFacePaddleCLas](https://github.com/Zheng-Bicheng/AdaFacePaddleCLas)。
仓库中可以利用AIStudio的免费GPU进行在线训练。


## 使用教程

**PC**
****

```python
from EasyDeploy.clas import AdaFaceForPC
import cv2
if __name__ == "__main__":
    model = AdaFaceForPC(verbose=True,
                         model_path="./weights/onnx/mobile_face_net_ada_face_112x112.onnx")
    model.export("./weights/rknn/mobile_face_net_ada_face_112x112.rknn")
    image = cv2.imread("./tests/test_outputs/scrfd_face.jpg")
    results = model.detect(image)
```

**Board**

```python
from EasyDeploy.clas import AdaFaceForBoard
import cv2
if __name__ == "__main__":
    model = AdaFaceForBoard(verbose=True,
                            rknn_path="./weights/rknn/mobile_face_net_ada_face_112x112.rknn")
    image = cv2.imread("./tests/test_outputs/scrfd_face.jpg")
    results = model.detect(image)
```

## 参考仓库🙏🙏🙏

原仓库地址: [AdaFace](https://github.com/mk-minchul/AdaFace)