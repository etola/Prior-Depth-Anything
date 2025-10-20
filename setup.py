from setuptools import setup, find_packages

setup(
    name="prior_depth_anything",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'torch==2.6.0',
        'torchvision==0.21.0',
        'numpy==2.2.6',
        'huggingface_hub==0.35.3',
        'einops==0.8.1',
        'Pillow==11.3.0',
        'opencv-python==4.12.0.88',
        'torch_cluster==1.6.3',
        'safetensors==0.6.2',
        'matplotlib==3.10.5'
    ],
    entry_points={
        "console_scripts": [
            "priorda = prior_depth_anything.cli:create_and_execute"
        ]
    },
    description="A pytorch implementation of Prior-Depth-Anything",
    author="Zehan Wang, Siyu Chen, Lihe Yang, Jialei Wang, Ziang Zhang, Hengshuang Zhao, Zhou Zhao",
    author_email="wangzehan01@zju.edu.cn",
    url="https://github.com/SpatialVision/Prior-Depth-Anything"
)