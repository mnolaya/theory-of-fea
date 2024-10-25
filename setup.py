from setuptools import setup, find_packages

setup(
    name='mfe',
    author='Michael N. Olaya',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'attrs',
        'polars',
        'ipykernel',
        'jupyter'
    ]
)