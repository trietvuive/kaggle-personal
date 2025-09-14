from setuptools import setup, find_packages

setup(
    name='kaggle_utils',
    version='0.1.0',
    description='A collection of useful Kaggle utility functions.',
    author='trietvuive',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'torch'
    ]
)