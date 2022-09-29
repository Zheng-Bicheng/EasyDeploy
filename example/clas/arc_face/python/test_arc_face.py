from EasyDeploy.clas import AdaFace
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
    model = AdaFace(verbose=config.verbose,
                    device=config.device,
                    model_path=config.model_path,
                    target_platform=config.target_platform)
    if config.device == 'pc':
        model.export("./arc_face.rknn")
    image = cv2.imread("./test_arcface.JPG")
    results = model.detect(image)
