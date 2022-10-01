from setuptools import setup, find_packages

setup(name='EasyDeploy',
      version='0.2.1',
      keywords=[],
      description='''
      An Easy and Fast Deployment Toolkit For RK3568.
      ''',
      license='Apache 2.0',
      packages=[
          'EasyDeploy',

          'EasyDeploy/base',

          'EasyDeploy/clas',
          'EasyDeploy/clas/ada_face',

          'EasyDeploy/detection',
          'EasyDeploy/detection/scrfd',
          'EasyDeploy/detection/picodet',

          'EasyDeploy/ocr',
          'EasyDeploy/ocr/ppocr',

          'EasyDeploy/project',
          'EasyDeploy/project/face_recognition',

          'EasyDeploy/segmentation',
          'EasyDeploy/segmentation/pp_humanseg',

          'EasyDeploy/utils',
      ],
      package_data={'': ['*.ttc']},
      data_files=[("", ["EasyDeploy/utils/simsun.ttc"])],
      author="ZhengBicheng",
      author_email="zheng_bicheng@outlook.com",
      url='https://github.com/Zheng-Bicheng/EasyDeployForRK3568',
      platforms='any',
      install_requires=[]
      )
