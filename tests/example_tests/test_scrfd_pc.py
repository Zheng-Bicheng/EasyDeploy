import os
from EaysDeploy.detection.scrfd import ScrFDForPC
import cv2
import numpy as np


def draw(img, bboxes, kpss, out_path, with_kps=True):
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox.astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if with_kps:
            if kpss is not None:
                kps = kpss[i].reshape(-1, 2)
                for kp in kps:
                    kp = kp.astype(np.int32)
                    cv2.circle(img, tuple(kp), 1, (255, 0, 0), 2)

    print('output:', out_path)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    my_model = ScrFDForPC(verbose=False,
                          model_path="./weights/onnx/scrfd_2.5g_bnkps_shape640x640.onnx")
    my_model.export("./weights/rknn")

    img = cv2.imread("./tests/test_images/ycy.jpeg")
    bboxes, landmarks = my_model.detect(img.copy())

    save_path = "./tests/test_outputs"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    draw(img, bboxes, landmarks, os.path.join(save_path,"scrfd_result.jpg"))
