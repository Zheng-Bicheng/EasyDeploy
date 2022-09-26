import sys
sys.path.append("./")
import os
from EasyDeploy.detection import SCRFDForBoard
from EasyDeploy.utils import draw_face
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
    draw_face(img.copy(), bboxes, landmarks, os.path.join(save_path, "scrfd_result.jpg"))
    print(landmarks[0].reshape(-1, 2))
    img_face = norm_crop(img.copy(), landmarks[0].reshape(-1, 2))
    cv2.imwrite(os.path.join(save_path, "scrfd_face.jpg"), img_face)
