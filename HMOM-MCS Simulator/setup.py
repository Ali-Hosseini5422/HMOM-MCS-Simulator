from setuptools import setup, find_packages

setup(
    name="hmom-mcs",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "networkx>=2.6",
        "tqdm>=4.62.0"
    ],
    python_requires=">=3.7",
)