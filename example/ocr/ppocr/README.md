# SCRFD Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

**运行PC代码，生成rknn文件**
```text
python  ./test_pp_ocr.py \
        --device pc \
        --model_path ./new_det.onnx \
        --target_platform RK3568
```

**board**
```text
sudo -E python  ./test_scrfd.py \
        --device board \
        --model_path ./scrfd.rknn \
        --target_platform RK3568
```
