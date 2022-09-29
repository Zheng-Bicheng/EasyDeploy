# PPHumanSeg Python部署示例

在部署前，需确认以下两个步骤:

* 正确安装EasyDeploy库
* 正确安装rknntoolkit库

执行如下脚本即可完成，部署测试:

* **运行PC代码，生成rknn文件**

    推理横屏图片
    ```text
    python  ./test_pp_humanseg.py \
            --device pc \
            --model_path ./new_heng.onnx \
            --target_platform RK3568
    ```
    推理竖屏图片
    ```text
    python  ./test_pp_humanseg.py \
            --device pc \
            --model_path ./new_shu.onnx \
            --target_platform RK3568
    ```

* **在板子上进行推理**
    ```text
    sudo -E python3  ./test_pp_humanseg.py \
                    --device board \
                    --model_path ./pp_humanseg.rknn \
                    --target_platform RK3568
    ```