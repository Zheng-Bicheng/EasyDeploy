from EasyDeploy.detection import SCRFD
from EasyDeploy.utils import draw_bbox
from EasyDeploy.utils import norm_crop
import cv2
import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=True, help="rknntoolkit verbose")
    parser.add_argument("--device", help="pc or board")
    parser.add_argument("--model_path", help="path of model for loading")
    parser.add_argument("--target_platform", help="rknntoolkit target_platform")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_config()
    model = SCRFD(verbose=config.verbose,
                  device=config.device,
                  model_path=config.model_path,
                  target_platform=config.target_platform)
    if config.device == 'pc':
        model.export("./scrfd.rknn")
    img = cv2.imread("./ada_face_test.jpeg")
    model.detect(img.copy())
    bboxes, landmarks = model.detect(img.copy())
    print("detect model ok")
    draw_bbox(img.copy(), bboxes, "./scrfd_result.jpg")
    img_face = norm_crop(img.copy(), landmarks[0].reshape(-1, 2))
    cv2.imwrite("./norm_crop.jpg", img_face)
