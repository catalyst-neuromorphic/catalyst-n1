from setuptools import setup, find_packages

setup(
    name="neurocore",
    version="1.0.0",
    description="Python SDK for the custom neuromorphic chip",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.5",
        "pyserial>=3.5",
    ],
    extras_require={
        "analysis": ["pandas>=1.4"],
    },
)
