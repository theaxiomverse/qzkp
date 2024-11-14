from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qzkp",
    version="0.1.0",
    author="Your Name",
    author_email="ncloutier@theaxiomverse.com",
    description="High-performance Quantum Zero-Knowledge Proof Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theaxiomverse/qzkp",
    project_urls={
        "Bug Tracker": "https://github.com/theaxiomverse/qzkp/issues",
        "Documentation": "https://github.com/theaxiomverse/qzkp/wiki",
        "Source Code": "https://github.com/theaxiomverse/qzkp",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-asyncio>=0.16.0",
            "pytest-benchmark>=3.4.1",
            "pytest-cov>=4.1.0",
            "black>=22.3.0",
            "isort>=5.10.1",
            "mypy>=0.950",
            "pylint>=2.14.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "qzkp=qzkp.cli:main",
        ],
    },
)
