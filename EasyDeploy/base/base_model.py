import os


def print_standard(standard_txt):
    print("Standard[EasyDeploy/base/base_model.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/base/base_model.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/base/base_model.py]: " + warning_txt)


class RKNNModel:
    def __init__(self,
                 verbose=True,
                 device=None):
        """
        初始化RKNNModel
        Args:
            verbose (bool):  是否开启日志
            device (str):  设备名，只能是pc或者board
        """
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        device = device.lower()
        if device == "pc":
            from rknn.api import RKNN
            self.model = RKNN(verbose)
        elif device == "board":
            from rknnlite.api import RKNNLite
            self.model = RKNNLite(verbose=verbose)
        self.device = device

        self.init = False

    def create_model(self,
                     mean_values=None,
                     std_values=None,
                     target_platform="RK3568",
                     model_path=None):
        """
        初始化 RKNN 或 RKNNLite
        Args:
            mean_values (list):  请参考rknntoolkit2的开发手册中的config参数
            std_values (list):  请参考rknntoolkit2的开发手册中的config参数
            target_platform (str):  请参考rknntoolkit2的开发手册中的target_platform参数
            model_path (str):  需要读取的模型地址
        """
        assert model_path is not None, print_error("model_path is None")

        if self.device.lower() == "pc":
            from rknn.api import RKNN
            # pre-process config
            if mean_values is None:
                print_warning("您没有配置mean_values，已为您初始化为[[0, 0, 0]]")
                print_warning("建议使用RKNN的config api进行标准化操作")
                mean_values = [[0, 0, 0]]
            if std_values is None:
                print_warning("您没有配置std_values，已为您初始化为[[0, 0, 0]]")
                print_warning("建议使用RKNN的config api进行标准化操作")
                std_values = [[1, 1, 1]]
            self.model.config(mean_values=mean_values, std_values=std_values, target_platform=target_platform)

            # Load ONNX model
            ret = self.model.load_onnx(model=model_path)
            assert ret == 0, print_error("Load model failed!")

            # Build model
            print_warning("EasyDeploy 不支持量化操作，如需量化，请参考rknn官方文档进行量化")
            ret = self.model.build(do_quantization=None)
            assert ret == 0, print_error("Build model failed!")

            # Init Runtime
            ret = self.model.init_runtime()
            assert ret == 0, print_error("Init runtime environment failed!")
        # end if self.device.lower() == "pc":
        elif self.device.lower() == "board":
            from rknnlite.api import RKNNLite
            # Load ONNX model
            ret = self.model.load_rknn(path=model_path)
            assert ret == 0, print_error("Load model failed!")

            # Init Runtime
            if target_platform == "RK3588":
                ret = self.model.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            else:
                ret = self.model.init_runtime()
            assert ret == 0, print_error("Init runtime environment failed!")
        # end elif self.device.lower() == "board":

        self.init = True

    def export(self, save_path="./weights/rknn/result.rknn"):
        """
        导出模型
        Args:
            save_path (str):  导出的地址
        """
        assert self.device == 'pc', print_error("只有在PC上运行才支持模型导出")
        assert self.init is True, print_error("请先初始化模型")

        # make save_dir
        export_path = os.path.dirname(save_path)
        if not os.path.exists(export_path):
            print_standard("保存目录不存在，已为您创建")
            os.mkdir(export_path)
        print_standard("导出的模型将保存在{}目录下".format(export_path))

        # export model
        ret = self.model.export_rknn(save_path)
        assert ret == 0, print_error("Export rknn model failed!")

    def infer(self, input_data):
        """
        推理，一般需要根据实际需求重构
        Args:
            input_data (list): 输入的数据
        Returns:
            result (list of numpy.ndarray): numpy数组
        """
        result = None
        if self.device == 'pc':
            result = self.model.inference(input_data)
        elif self.device == 'board':
            result = self.model.inference(input_data)
        return result
