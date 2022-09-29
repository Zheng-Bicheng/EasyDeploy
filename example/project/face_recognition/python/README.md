# Face Recognition

## 使用教程

**运行PC代码，生成rknn文件**
```text
wget https://bj.bcebos.com/fastdeploy/models/onnx/mobile_face_net_ada_face_112x112.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/scrfd_2.5g_bnkps_shape640x640.onnx
python  ./test_face_recognition.py \
        --device pc \
        --det_model_name scrfd \
        --det_model_weight_path ./scrfd_2.5g_bnkps_shape640x640.onnx \
        --cls_model_name adaface \
        --cls_model_weight_path ./mobile_face_net_ada_face_112x112.onnx \
        --target_platform RK3568
```

**board**
```text
sudo -E python3  ./test_face_recognition.py \
                --device board \
                --det_model_name scrfd \
                --det_model_weight_path ./det.rknn \
                --cls_model_name adaface \
                --cls_model_weight_path ./cls.rknn \
                --target_platform RK3568
```
