# PP-HumanSeg

## 模型介绍
将人物和背景在像素级别进行区分，是一个图像分割的经典任务，具有广泛的应用。 一般而言，该任务可以分为两类：针对半身人像的分割，简称肖像分割；针对全身和半身人像的分割，简称通用人像分割。

对于肖像分割和通用人像分割，PaddleSeg发布了PP-HumanSeg系列模型，具有分割精度高、推理速度快、通用型强的优点。而且PP-HumanSeg系列模型可以开箱即用，零成本部署到产品中，也支持针对特定场景数据进行微调，实现更佳分割效果。

## 下载预训练ONNX模型
* 从百度网盘进行下载
  ```text
    链接:https://pan.baidu.com/s/1dUqf9y6pMfWAHMma40xY6w?pwd=zzbc 提取码:zzbc 复制这段内容后打开百度网盘手机App，操作更方便哦
  ```
* 从AIStudio对模型进行转换并下载
   ```text
    我发现了一篇高质量的实训项目，使用免费算力即可一键运行，还能额外获取8小时免费GPU运行时长，快来Fork一下体验吧。
    模型集市——Paddle系列模型ONNX合集：https://aistudio.baidu.com/aistudio/projectdetail/4618218?contributionType=1&sUid=790375&shared=1&ts=1664425117252
  ```

## 详细部署文档

- [Python部署](python)

## 参考仓库🙏🙏🙏

原仓库地址: [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/README_cn.md)
