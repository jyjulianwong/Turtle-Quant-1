from setuptools import find_packages, setup

setup(
    name="turtle-quant-1",
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
