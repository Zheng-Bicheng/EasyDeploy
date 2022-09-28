# SCRFD Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

**运行PC代码，生成rknn文件**
```text
wget https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape640x640.onnx
python  ./test_scrfd_test.py \
        --device pc \
        --model_path ./scrfd_2.5g_bnkps_shape640x640.onnx \
        --target_platform RK3568
```

**board**
```text
python  ./test_scrfd_test.py \
        --device board \
        --model_path ./scrfd.rknn \
        --target_platform RK3568
```
