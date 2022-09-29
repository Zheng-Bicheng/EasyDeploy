from EasyDeploy.detection import Picodet
import cv2
import argparse

label_list = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv monitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


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
    model = Picodet(verbose=config.verbose,
                    device=config.device,
                    model_path=config.model_path,
                    target_platform=config.target_platform)
    if config.device == 'pc':
        model.export("./picodet.rknn")
    img = cv2.imread("./picodet_demo_input.jpg")
    output_image = model.detect(img.copy(), label_list)
    print("detect model ok")
    cv2.imwrite("./picodet_result.jpg", output_image)
