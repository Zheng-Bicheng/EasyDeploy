# EasyDeployForRK3568

## 简介
****
**EasyDeployForRK3568**是针对柠檬派RK3568开发的推理部署工具箱。EasyDeploy集成了多种AI模型，能快速的在柠檬派RK3568上部署各种例程，
做到开箱即用，大大缩减了工程所需要的时间，满足开发者多场景的产业部署需求。

## 项目结构介绍
****
```text
.
├── EasyDeploy  python源代码目录
├── LICENSE
├── README.md
├── docs  存放各种文档
├── setup.py
├── tests  测试代码
└── weights  权重文件文件夹
```

## 模型支持列表✨✨✨
****

符号说明: (1) ✅: 已经支持  (2) ❌: 暂不支持

| 任务场景           | 模型      | API                                                                                                 | 是否支持量化 | RK3568 | RK3588 |
|----------------|---------|-----------------------------------------------------------------------------------------------------|--------|--------|--------|
| Face Clas      | AdaFace | [python](https://github.com/Zheng-Bicheng/EasyDeployForRK3568/tree/main/EasyDeploy/clas/ada_face)   | ❌      | ✅      | ✅      |
| Face Detection | ScrFD   | [python](https://github.com/Zheng-Bicheng/EasyDeployForRK3568/tree/main/EasyDeploy/detection/scrfd) | ❌      | ✅      | ✅      |

## demo例程✨✨✨
| 任务场景             | API                                                                                                           | 是否支持量化 | RK3568 | RK3588 |
|------------------|---------------------------------------------------------------------------------------------------------------|--------|--------|--------|
| Face Recognition | [python](https://github.com/Zheng-Bicheng/EasyDeployForRK3568/tree/main/tests/project_tests/face_recognition) | ❌      | ✅      | ✅      |


## 文档导航🚙🚙🚙
****

## 联系我📮📮📮
****

我的邮箱: [zheng_bicheng@outlook.com](zheng_bicheng@outlook.com)