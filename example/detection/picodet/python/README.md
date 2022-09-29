# Picodet Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

**运行PC代码，生成rknn文件**
```text
wget wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/picodet_s_320_coco_sim.onnx
python  ./test_picodet.py \
        --device pc \
        --model_path ./picodet_s_320_coco_sim.onnx \
        --target_platform RK3568
```

**board**
```text
python  ./test_picodet.py \
        --device board \
        --model_path ./picodet.rknn \
        --target_platform RK3568
```
