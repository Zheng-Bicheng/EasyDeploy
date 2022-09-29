from EasyDeploy.segmentation.pp_humanseg import PPHumanSeg
import argparse
import cv2


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
    model = PPHumanSeg(verbose=config.verbose,
                       device=config.device,
                       model_path=config.model_path,
                       target_platform=config.target_platform)
    if config.device == 'pc':
        model.export("./pp_humanseg.rknn")

    img = cv2.imread("./scrfd_test.jpeg")
    model.detect(img.copy())
    output_image = model.detect(img.copy())
    print("detect model ok")
    cv2.imwrite("./output.jpg", output_image)