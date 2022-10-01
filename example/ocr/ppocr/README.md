# SCRFD Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

**运行PC代码，生成rknn文件**
```text
python  ./test_pp_ocr.py \
        --device pc \
        --det_model_path ./ppocrv3_chinese_english/new_det.onnx \
        --rec_model_path ./ppocrv3_chinese_english/new_rec.onnx \
        --rec_char_dict_path ./ppocrv3_chinese_english/ppocr_keys_v1.txt \
        --target_platform RK3568
        
        
python  ./test_pp_ocr.py \
        --device pc \
        --det_model_path ./PP_OCR_v2_det.onnx \
        --rec_model_path ./PP_OCR_v2_rec.onnx \
        --rec_char_dict_path ./ppocrv3_chinese_english/ppocr_keys_v1.txt \
        --target_platform RK3568       
```

**board**
```text
sudo -E python  ./test_scrfd.py \
        --device board \
        --model_path ./scrfd.rknn \
        --target_platform RK3568
```
