# SCRFD
SCRFD是一种高效的轻量级高精度人脸检测方法，被ICLR-2022接受。

## 精度

| Name          | Easy | Medium | Hard | FLOPs | Params(M) | Infer(ms)  | 
|---------------|-----|------|-----|-----|---------|------------| 
| SCRFD_500M    | 90.57 | 88.12 | 68.51 | 500M | 0.57    | 3.6        | 
| SCRFD_1G      | 92.38 | 90.57 | 74.80 | 1G  | 0.64    | 4.1        |
| SCRFD_2.5G    | 93.78 | 92.16 | 77.87 | 2.5G | 0.67    | 4.2        | 
| SCRFD_10G     | 95.16 | 93.87 | 83.05 | 10G | 3.86    | 4.9        | 
| SCRFD_34G     | 96.06 | 94.92 | 85.29 | 34G | 9.80    | 11.7       | 
| SCRFD_500M_KPS | 90.97 | 88.44 | 69.49 | 500M | 0.57    | 3.6        |
| SCRFD_2.5G_KPS | 93.80 | 92.02 | 77.13 | 2.5G | 0.82    | 4.3        |
| SCRFD_10G_KPS | 95.40 | 94.01 | 82.80 | 10G | 4.23    | 5.0        |

## 使用教程

注意：如果需要推理人脸关键点，请使用带关键点的模型，即百度网盘中的with_kps文件夹下的模型

**PC**
****

```python
import os
from EasyDeploy.detection import SCRFDForPC
from EasyDeploy.utils import draw_bbox
from EasyDeploy.utils import norm_crop
import cv2

if __name__ == "__main__":
    my_model = SCRFDForPC(verbose=False,
                          model_path="./weights/onnx/scrfd_640x640_with_points.onnx")
    my_model.export("./weights/rknn/scrfd_640x640_with_points.rknn")

    img = cv2.imread("./tests/test_images/ada_face_test.jpeg")
    bboxes, landmarks = my_model.detect(img.copy())

    save_path = "./tests/test_outputs/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    draw_bbox(img.copy(), bboxes, landmarks, os.path.join(save_path, "scrfd_result.jpg"))

    img_face = norm_crop(img.copy(), landmarks[0])
    cv2.imwrite(os.path.join(save_path, "scrfd_face.jpg"), img_face)
```

**board**
****

```python
import sys

sys.path.append("./")
import os
from EasyDeploy.detection import SCRFDForBoard
from EasyDeploy.utils import draw_bbox
from EasyDeploy.utils import norm_crop
import cv2

if __name__ == "__main__":
    my_model = SCRFDForBoard(verbose=False,
                             rknn_path="./weights/rknn/scrfd_640x640_with_points.rknn")
    img = cv2.imread("./tests/test_images/ada_face_test.jpeg")
    my_model.detect(img.copy())
    bboxes, landmarks = my_model.detect(img.copy())
    print("detect model ok")
    save_path = "./tests/test_outputs/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    draw_bbox(img.copy(), bboxes, landmarks, os.path.join(save_path, "scrfd_result.jpg"))
    print(landmarks[0].reshape(-1, 2))
    img_face = norm_crop(img.copy(), landmarks[0].reshape(-1, 2))
    cv2.imwrite(os.path.join(save_path, "scrfd_face.jpg"), img_face)
```

## 参考仓库🙏🙏🙏

原仓库地址: [insightface](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

代码推理参考: [FastDeploy](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/facedet/scrfd)