"""
Setup file for SC2 package.
This is a minimal setup.py for compatibility with older pip versions.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="SC2",
    version="0.1.0",
    packages=find_packages(include=["SC2", "SC2.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
    ],
)