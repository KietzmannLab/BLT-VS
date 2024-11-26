from setuptools import setup, find_packages

setup(
    name='blt_vs_model',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='BLT_VS model package',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'huggingface_hub>=0.15.1',
        'Pillow>=9.0.0',
        'importlib_resources>=5.1.0; python_version<"3.9"',
        'matplotlib>=3.4.0',
        'requests>=2.25.0'
    ],
    package_data={
        'blt_vs_model': ['imagenet.json', 'ecoset.json'],
    },
)