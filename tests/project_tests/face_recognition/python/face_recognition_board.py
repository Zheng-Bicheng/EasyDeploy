# encoding: utf-8
from EasyDeploy.project.face_recognition import FaceRecognitionForBoard
from EasyDeploy.utils.face_detection import draw_face_with_text
import cv2
import os

if __name__ == "__main__":
    face_recognition = FaceRecognitionForBoard(det_model_name="scrfd",
                                               det_model_weight_path="./weights/det.rknn",
                                               cls_model_name="adaface",
                                               cls_model_weight_path="./weights/cls.rknn")
    face_recognition.create_database("./database", "./database")
    face_recognition.load_database("./database")

    pictures_path = "./test_images"
    label_idx_list = ['ml', 'xx', 'zjk']

    for picture_name in os.listdir(pictures_path):
        picture_path = os.path.join(pictures_path, picture_name)
        img = cv2.imread(picture_path)
        results, bboxes = face_recognition.predict(img)
        print(results)
        if results is None:
            print("未检测出人脸")
            continue

        img = draw_face_with_text(img.copy(), bboxes, results, label_idx_list)
        cv2.imwrite(os.path.join("./test_outputs", picture_name), img)
