import cv2
from EasyDeploy.utils import (det_resize, pad_stride)
from EasyDeploy.base import RKNNModel
import numpy as np


def print_standard(standard_txt):
    print("Standard[EasyDeploy/ocr/pp_ocr/pp_ocr_rec.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/ocr/pp_ocr/pp_ocr_rec.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/ocr/pp_ocr/pp_ocr_rec.py]: " + warning_txt)


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                        batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                                                        idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple):
            preds = preds[-1]
        # if isinstance(preds, paddle.Tensor):
        #     preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class PPOCRRec(RKNNModel):
    def __init__(self,
                 verbose=True,
                 device=None,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None,
                 input_size=None,
                 rec_char_dict_path=None,
                 use_space_char=None):
        # config device
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        super(PPOCRRec, self).__init__(
            verbose=verbose,
            device=device
        )

        # create model
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.5, 0.5, 0.5]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.5, 0.5, 0.5]]]
        assert model_path is not None, print_error("model_path is None")
        self.create_model(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path)

        if input_size is None:
            input_size = [48, 320]
        self.input_size = input_size

        assert rec_char_dict_path is not None, print_error("rec_char_dict_path is None")
        assert use_space_char is not None, print_error("use_space_char is None")
        self.postprocess_op = CTCLabelDecode(rec_char_dict_path,
                                             use_space_char)

    def detect(self,
               img_list):
        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        for beg_img_no, img in enumerate(img_list):
            input_img = cv2.resize(img.copy(), self.input_size[::-1])
            input_img = np.expand_dims(input_img.copy(), axis=0)
            results = self.infer([input_img])
            preds = results[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res
