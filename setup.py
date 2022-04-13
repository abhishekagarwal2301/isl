from setuptools import find_packages, setup

setup(
    name="isl",
    version="1.0.0",
    author="Abhishek Agarwal",
    author_email="abhishek.agarwal@npl.co.uk",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/abhishekagarwal2301/isl",
    license="LICENSE",
    description="A package for implementing the Incremental \
        Structure Learning (ISL) algorithm for approximate \
            circuit recompilation",
    long_description=open("README.md").read(),
    install_requires=[
        "qiskit",
        "numpy",
        "scipy",
        "openfermion",
    ],
)
