from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=2.4.1', 'numpy>=1.19.5', 'transformers>=4.6.1', 'sentence-transformers>=1.2.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Susubert trainer and matcher'
)