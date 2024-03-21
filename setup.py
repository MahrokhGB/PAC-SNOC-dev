#!/usr/bin/env python
import setuptools
import os

# os.chmod("run.py", 0o744)

setuptools.setup(
    name='Pac-SNOC',
    version='1.0',
    url='https://github.com/DecodEPFL/PAC-SNOC',
    license='CC-BY-4.0 License',
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.7.1',
                      'numpy>=1.18.1',
                      'matplotlib>=3.1.3'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)
