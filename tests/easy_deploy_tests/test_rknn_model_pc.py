from EaysDeploy.base.base_model import RKNNModelPC

if __name__ == "__main__":
    # 测试模型是否能够成功构建
    my_model = RKNNModelPC(model_path="./weights/onnx/scrfd_2.5g_bnkps_shape640x640.onnx")
    # 测试模型是否能够成功导出
    my_model.export("./weights/rknn")
