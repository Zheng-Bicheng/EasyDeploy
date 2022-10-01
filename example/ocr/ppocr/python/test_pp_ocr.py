from EasyDeploy.ocr import pp_ocr_det
from EasyDeploy.ocr import PPOCRRec
from EasyDeploy.ocr import PPOCRCls

import cv2
import argparse


def postprocess(dt_boxes, rec_res, drop_score=0.85):
    filter_boxes, filter_rec_res = [], []
    for box, rec_result in zip(dt_boxes, rec_res):
        text, score = rec_result
        if score >= drop_score:
            filter_boxes.append(box)
            filter_rec_res.append(rec_result)
    return filter_boxes, filter_rec_res


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=False, help="rknntoolkit verbose")
    parser.add_argument("--device", help="pc or board")
    parser.add_argument("--det_model_path", help="path of model for loading")
    parser.add_argument("--cls_model_path", default=None, help="path of model for loading")
    parser.add_argument("--rec_model_path", default=None, help="path of model for loading")
    parser.add_argument("--rec_char_dict_path", default=None, help="path of model for loading")
    parser.add_argument("--target_platform", help="rknntoolkit target_platform")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_config()
    # 加载模型
    det_model = pp_ocr_det.PPOCRDet(verbose=config.verbose,
                                    device=config.device,
                                    model_path=config.det_model_path,
                                    target_platform=config.target_platform,
                                    thresh=0.8,
                                    box_thresh=0.5)

    # 导出模型
    if config.device == 'pc':
        det_model.export("./det.rknn")

    # 读取图片
    img = cv2.imread("./pp_ocr_demo.png")

    # 对模型进行推理
    bboxes, img_crop_list, output_image = det_model.detect(img.copy())
    print("detect model ok")

    # 保存图片，可以不进行这部分,这里设置为如果不rec则保存det结果
    if config.rec_model_path is not None:
        output_image = pp_ocr_det.draw_det(output_image, bboxes)
        cv2.imwrite("./result.png", output_image)

    # ppocr_cls
    if config.cls_model_path is not None:
        cls_model = PPOCRCls(verbose=config.verbose,
                             device=config.device,
                             model_path=config.cls_model_path,
                             target_platform=config.target_platform)
        img_crop_list, angle_list = cls_model.detect(img_list=img_crop_list)
        if config.device == 'pc':
            cls_model.export("./cls.rknn")

    # ppocr_rec
    if config.rec_model_path is not None:
        assert config.rec_char_dict_path is not None, print("rec_char_dict_path is None")
        rec_model = PPOCRRec(verbose=config.verbose,
                             device=config.device,
                             model_path=config.rec_model_path,
                             target_platform=config.target_platform,
                             rec_char_dict_path=config.rec_char_dict_path,
                             use_space_char=True)
        if config.device == 'pc':
            rec_model.export("./rec.rknn")
        filter_rec_res = rec_model.detect(img_crop_list)
        _, filter_rec_res = postprocess(bboxes, filter_rec_res)
        for text, score in filter_rec_res:
            print("{}, {:.3f}".format(text, score))
        print("Finish!")
