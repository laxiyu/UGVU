"""UGVU: Uncertainty-Guided Generative Vision Understanding via Consensus Reasoning.

Towards Reliable Dense Prediction from Black-Box Image Generators.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ugvu",
    version="1.0.0",
    description="Uncertainty-Guided Generative Vision Understanding via Consensus Reasoning",
    author="UGVU Research Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ugvu=ugvu.__main__:cli",
        ],
    },
)
