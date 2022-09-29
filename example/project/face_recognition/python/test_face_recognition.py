# encoding: utf-8
from EasyDeploy.project.face_recognition import FaceRecognition
from EasyDeploy.utils import draw_face_with_text
import cv2
import os
import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=False, help="rknntoolkit verbose")
    parser.add_argument("--device", help="pc or board")
    parser.add_argument("--det_model_name", help="name of det model for loading")
    parser.add_argument("--det_model_weight_path", help="path of det model for loading")
    parser.add_argument("--cls_model_name", help="name of cls model for loading")
    parser.add_argument("--cls_model_weight_path", help="path of cls model for loading")
    parser.add_argument("--target_platform", help="rknntoolkit target_platform")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_config()
    face_recognition = FaceRecognition(verbose=config.verbose,
                                       device=config.device,
                                       target_platform=config.target_platform,
                                       det_model_name=config.det_model_name,
                                       det_model_weight_path=config.det_model_weight_path,
                                       cls_model_name=config.cls_model_name,
                                       cls_model_weight_path=config.cls_model_weight_path
                                       )
    if config.device == 'pc':
        face_recognition.export("./")
    face_recognition.create_database("./database", "./database")
    face_recognition.load_database("./database")

    pictures_path = "./test_images"
    label_idx_list = ['ml', 'xx', 'zjk']

    for picture_name in os.listdir(pictures_path):
        picture_path = os.path.join(pictures_path, picture_name)
        img = cv2.imread(picture_path)
        results, bboxes = face_recognition.detect(img)
        if results is None:
            print("未检测出人脸")
            continue
        img = draw_face_with_text(img.copy(), bboxes, results, label_idx_list)
        cv2.imwrite(os.path.join("./test_outputs", picture_name), img)
