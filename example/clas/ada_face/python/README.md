# AdaFace Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

**运行PC代码，生成rknn文件**
```text
wget https://bj.bcebos.com/fastdeploy/models/onnx/mobile_face_net_ada_face_112x112.onnx
python  ./test_ada_face.py \
        --device pc \
        --model_path ./mobile_face_net_ada_face_112x112.onnx \
        --target_platform RK3568
```

**board**
```text
python  ./test_ada_face.py \
        --device board \
        --model_path ./ada_face.rknn \
        --target_platform RK3568
```
