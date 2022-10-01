from EasyDeploy.base import RKNNModel
import cv2
import numpy as np
import math


def print_standard(standard_txt):
    print("Standard[EasyDeploy/ocr/pp_ocr/pp_ocr_cls.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/ocr/pp_ocr/pp_ocr_cls.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/ocr/pp_ocr/pp_ocr_cls.py]: " + warning_txt)


class PPOCRCls(RKNNModel):
    def __init__(self,
                 verbose=True,
                 device=None,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None,
                 input_size=None,
                 cls_thresh=0.5):
        # config device
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        super(PPOCRCls, self).__init__(
            verbose=verbose,
            device=device
        )

        # create model
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.485, 0.456, 0.406]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.229, 0.224, 0.225]]]
        assert model_path is not None, print_error("model_path is None")
        self.create_model(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path)

        if input_size is None:
            self.input_size = [48, 320]

        self.postprocess_op = ClsPostProcess(['0', '180'])
        self.cls_thresh = cls_thresh

    def detect(self, img_list):
        img_num = len(img_list)
        cls_res = [['', 0.0]] * img_num
        for beg_img_no, img in enumerate(img_list):
            img = self.resize_norm_img(img.copy())
            img = np.expand_dims(img, axis=0)
            outputs = self.infer([img])[0]
            # print(outputs.shape)
            # outputs = np.argmax(outputs, axis=0)
            # prob_out = outputs
            cls_result = self.postprocess_op(outputs)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[beg_img_no + rno] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[beg_img_no + rno] = cv2.rotate(img_list[beg_img_no + rno], 1)
        return img_list, cls_res

    def resize_norm_img(self, img):
        imgC, [imgH, imgW] = 3, self.input_size
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
        padding_im[:, 0:resized_w, :] = resized_image
        return padding_im


class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None, *args, **kwargs):
        # if isinstance(preds, paddle.Tensor):
        #     preds = preds.numpy()
        pred_idxs = preds.argmax(axis=1)
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = [(self.label_list[idx], 1.0) for idx in label]
        return decode_out, label
