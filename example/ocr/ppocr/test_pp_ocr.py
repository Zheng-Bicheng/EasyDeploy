from EasyDeploy.ocr import pp_ocr_det
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
    # 加载模型
    det_model = pp_ocr_det.PPOCRDet(verbose=config.verbose,
                                    device=config.device,
                                    model_path=config.model_path,
                                    target_platform=config.target_platform,
                                    thresh=0.7,
                                    box_thresh=0.7)

    # 导出模型
    if config.device == 'pc':
        det_model.export("./det.rknn")

    # 读取图片
    img = cv2.imread("./pp_ocr_demo.png")

    # 对模型进行推理
    bboxes, landmarks = det_model.detect(img.copy())
    print("detect model ok")

    # 保存图片，可以不进行这部分
    img = pp_ocr_det.draw_det(img.copy(), bboxes)
    cv2.imwrite("./result.png", img)
