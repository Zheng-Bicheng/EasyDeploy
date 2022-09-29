# ArcFace Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

**运行PC代码，生成rknn文件**
```text
wget https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx
python  ./test_arc_face.py \
        --device pc \
        --model_path ./ms1mv3_arcface_r100.onnx \
        --target_platform RK3568
```

**board**
```text
python  ./test_arc_face.py \
        --device board \
        --model_path ./arc_face.rknn \
        --target_platform RK3568
```
