from EasyDeploy.project.face_recognition import FaceRecognitionForPC
import cv2
import os

if __name__ == "__main__":
    face_recognition = FaceRecognitionForPC(det_model_name="scrfd",
                                            det_model_weight_path="./weights/onnx/scrfd_640x640_with_points.onnx",
                                            cls_model_name="adaface",
                                            cls_model_weight_path="./weights/onnx/mobile_face_net_ada_face_112x112.onnx")
    face_recognition.export("./weights")
    face_recognition.create_database("./database", "./database")
    face_recognition.load_database("./database")

    pictures_path = "./test_images"
    for picture_name in os.listdir(pictures_path):
        picture_path = os.path.join(pictures_path, picture_name)
        img = cv2.imread(picture_path)
        results = face_recognition.predict(img)
        print(results)
        if results is None:
            print("未检测出人脸")
            continue
        for result in results:
            print("{} 是 {} 的概率为{}".format(picture_name, result[0][0], result[0][1]))
