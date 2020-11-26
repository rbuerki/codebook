import setuptools
from pathlib import Path

from src import __version__ as version

readme = Path.cwd() / "README.md"
with open(readme, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="raph-codebook",
    version=version,
    author="Raphael Bürki",
    author_email="r2d4@bluewin.ch",
    description="A library of helpful functions and classes for my projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rbuerki/codebook/",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "seaborn",
    ],
)
