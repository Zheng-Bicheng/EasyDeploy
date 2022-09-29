from EasyDeploy.base import RKNNModel
import numpy as np


def print_standard(standard_txt):
    print("Standard[EasyDeploy/clas/arc_face/arc_face.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/clas/arc_face/arc_face.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/clas/arc_face/arc_face.py]: " + warning_txt)


class ArcFace(RKNNModel):
    def __init__(self,
                 verbose=True,
                 device=None,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None):
        # config device
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        super(ArcFace, self).__init__(
            verbose=verbose,
            device=device
        )

        # create model
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.5, 0.5, 0.5]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.5, 0.5, 0.5]]]
        assert model_path is not None, print_error("model_path is None")
        assert model_path is not None, print_error("model_path is None")
        self.create_model(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path)

    def detect(self, img):
        input_data = np.array(img).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        results = self.infer([input_data])[0]
        return results