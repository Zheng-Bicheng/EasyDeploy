# EasyDeployForRK3568

## 简介
****
**EasyDeploy**是针对凌蒙派RK3568开发的推理部署工具箱。EasyDeploy集成了多种AI模型，能快速的在凌蒙派RK3568上部署各种例程，
做到开箱即用，大大缩减了工程所需要的时间，满足开发者多场景的产业部署需求。

## 模型支持列表✨✨✨
****

符号说明: (1) ✅: 已经支持  (2) ❌: 暂不支持

| 任务场景                | 模型         | Demo                                        | 是否支持量化 | RK3568 | RK3588 |
|---------------------|------------|---------------------------------------------|--------|--------|--------|
| Face Clas           | AdaFace    | [Demo](./example/clas/ada_face/)            | ❌      | ✅      | ✅      |
| Face Clas           | ArcFace    | [Demo](./example/clas/arc_face/)            | ❌      | ✅      | ✅      |
| Face Detection      | ScrFD      | [Demo](./example/detection/scrfd/)          | ❌      | ✅      | ✅      |
| Universal Detection | Picodet    | [Demo](./example/detection/picodet/)        | ❌      | ✅      | ✅      |
| People Segmentation | PPHumanSeg | [Demo](./example/segmentation/pp_humanseg/) | ❌      | ✅      | ✅      |

## demo例程✨✨✨
| 任务场景           | Demo                                                | 是否支持量化 | RK3568 | RK3588 |
|----------------|-----------------------------------------------------|--------|--------|--------|
| 人脸识别           | [python](./example/project/face_recognition/python) | ❌      | ✅      | ✅      |
| PPHumanSeg人像分割 | [python](./example/segmentation/pp_humanseg/python) | ❌      | ✅      | ✅      |
| Picodet 通用检测模型 | [python](./example/detection/picodet/python)        | ❌      | ✅      | ✅      |


## 文档导航🚙🚙🚙
****

* [EasyDeploy安装手册](./docs/quickly_start/install.md)

## 联系我📮📮📮
****

我的邮箱: [zheng_bicheng@outlook.com](zheng_bicheng@outlook.com)