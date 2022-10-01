# PPOCR Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

## 运行代码

**运行PC代码，生成rknn文件**

```text
python  ./test_pp_ocr.py \
        --device pc \
        --det_model_path ./ppocrv3_chinese_english/new_det.onnx \
        --cls_model_path ./ppocrv3_chinese_english/new_cls.onnx \
        --rec_model_path ./ppocrv3_chinese_english/new_rec.onnx \
        --rec_char_dict_path ./ppocrv3_chinese_english/ppocr_keys_v1.txt \
        --target_platform RK3568
        
```

**board**

```text
sudo -E python3  ./test_pp_ocr.py \
        --device board \
        --det_model_path ./det.rknn \
        --cls_model_path ./cls.rknn \
        --rec_model_path ./rec.rknn \
        --rec_char_dict_path ./ppocrv3_chinese_english/ppocr_keys_v1.txt \
        --target_platform RK3568
```
