from setuptools import setup, find_packages

setup(name='EasyDeploy',
      version='0.0.1',
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
          'EasyDeploy/utils',
      ],
      author="ZhengBicheng",
      author_email="zheng_bicheng@outlook.com",
      url='https://github.com/Zheng-Bicheng/EasyDeployForRK3568',
      platforms='any',
      install_requires=[]
      )
