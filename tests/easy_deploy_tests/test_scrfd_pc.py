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

    img_face = norm_crop(img.copy(),landmarks[0])
    cv2.imwrite(os.path.join(save_path, "scrfd_face.jpg"), img_face)