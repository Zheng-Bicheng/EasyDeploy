from EaysDeploy.clas.ada_face.ada_face import AdaFaceForPC
import cv2
if __name__ == "__main__":
    model = AdaFaceForPC(verbose=True,
                         model_path="./weights/onnx/mobile_face_net_ada_face_112x112.onnx")
    model.export("./weights/rknn/mobile_face_net_ada_face_112x112.rknn")
    image = cv2.imread("./tests/test_outputs/scrfd_face.jpg")
    results = model.detect(image)
    print(results[0].shape)