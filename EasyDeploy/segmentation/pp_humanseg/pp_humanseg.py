from EasyDeploy.base import RKNNModel
import cv2
import numpy as np


def print_standard(standard_txt):
    print("Standard[EasyDeploy/segmentation/pp_humanseg/pp_humanseg.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/segmentation/pp_humanseg/pp_humanseg.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/segmentation/pp_humanseg/pp_humanseg.py]: " + warning_txt)


def resize(im, target_size, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        h = target_size[0]
        w = target_size[1]
    else:
        h = target_size
        w = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def display_masked_image(mask, image, color_map=None, weight=0.6):
    if color_map is None:
        color_map = [255, 0, 0]
    mask = mask > 0
    c1 = np.zeros(shape=mask.shape, dtype='uint8')
    c2 = np.zeros(shape=mask.shape, dtype='uint8')
    c3 = np.zeros(shape=mask.shape, dtype='uint8')
    pseudo_img = np.dstack((c1, c2, c3))
    for i in range(3):
        pseudo_img[:, :, i][mask] = color_map[i]
    vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
    return vis_result


class PPHumanSeg(RKNNModel):
    def __init__(self,
                 verbose=True,
                 device=None,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None,
                 input_size=None):
        # config device
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        super(PPHumanSeg, self).__init__(
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
            # input_size = [h,w]
            self.input_size = [144, 256]

    def detect(self, image):
        im = resize(image, self.input_size)
        result = self.infer([im])[0][0]
        pred = np.argmax(result, axis=0)

        raw_frame = display_masked_image(pred, im)
        raw_frame = resize(raw_frame, target_size=image.shape[0:2])
        return raw_frame
