from setuptools import setup, find_packages

setup(
    name="minitst",
    version="0.1.0",
    description="3-D molecule geometry and conformer handling (species + geometry modules from AutoTST)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AutoTST Team",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "ase",
        "rdkit",
    ],
    extras_require={
        "viz": ["py3Dmol"],
    },
)
