import os


def print_error(error_txt):
    print("Error[base/base_model.py]: " + error_txt)


def print_warning(error_txt):
    print("Warning[base/base_model.py]: " + error_txt)


class RKNNModelPC:
    def __init__(self,
                 verbose=True,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None):
        from rknn.api import RKNN

        # create rknn
        self.model = RKNN(verbose)

        # pre-process config
        if mean_values is None:
            print_warning("您没有配置mean_values，已为您初始化为[[0, 0, 0]]")
            print_warning("建议使用RKNN的config api进行标准化操作")
            mean_values = [[0, 0, 0]]
        if std_values is None:
            print_warning("您没有配置std_values，已为您初始化为[[0, 0, 0]]")
            print_warning("建议使用RKNN的config api进行标准化操作")
            std_values = [[1, 1, 1]]
        if target_platform is None:
            target_platform = "RK3568"
        self.model.config(mean_values=mean_values, std_values=std_values, target_platform=target_platform)

        # Load ONNX model
        assert model_path is not None, print_error("model_path is None")
        ret = self.model.load_onnx(model=model_path)
        assert ret == 0, print_error("Load model failed!")

        # Build model
        print_warning("EasyDeploy 不支持量化操作，如需量化，请参考rknn官方文档进行量化")
        ret = self.model.build(do_quantization=None)
        assert ret == 0, print_error("Build model failed!")

        # Init Runtime
        ret = self.model.init_runtime()
        assert ret == 0, print_error("Init runtime environment failed!")

    def export(self, save_path="./weights/rknn/result.rknn"):
        export_path = os.path.dirname(save_path)
        if not os.path.exists(export_path):
            os.mkdir(export_path)
        print("导出的模型将保存在{}目录下".format(export_path))
        ret = self.model.export_rknn(save_path)
        assert ret == 0, print_error("Export rknn model failed!")

    def infer(self, input_data):
        result = self.model.inference(input_data)
        return result
