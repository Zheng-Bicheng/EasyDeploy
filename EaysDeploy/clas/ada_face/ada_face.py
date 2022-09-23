from EaysDeploy.base import RKNNModelPC
import numpy as np


class AdaFaceForPC(RKNNModelPC):
    def __init__(self,
                 verbose=True,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None):
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.5, 0.5, 0.5]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.5, 0.5, 0.5]]]
        super(AdaFaceForPC, self).__init__(
            verbose=verbose,
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path
        )

    def detect(self, img):
        input_data = np.array(img).astype(np.float32)
        results = self.infer([input_data])
        return results