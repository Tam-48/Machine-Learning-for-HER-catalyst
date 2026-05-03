"""
Setup.py for ML-guided bimetallic catalyst design package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-bimetallic-catalyst-her",
    version="0.1.0",
    author="Research Team",
    author_email="contact@example.com",
    description="ML framework for designing optimal bimetallic catalysts for HER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tam-48/Machine-Learning-for-HER-catalyst",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.0.1",
        "tensorflow>=2.13.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "ase>=3.23.0",
        "pydantic>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
        ],
        "ocp": [
            "fairchem-core>=0.3.0",
            "wandb>=0.15.8",
        ],
    },
)
