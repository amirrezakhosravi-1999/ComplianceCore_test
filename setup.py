#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for CAELUS - Compliance Assessment Engine Leveraging Unified Semantics
"""

from setuptools import setup, find_packages
import os

# Read the README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, 'README_EN.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(current_dir, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="caelus-compliance",
    version="1.0.0",
    author="Parscoders Team",
    author_email="info@parscoders.com",
    description="AI-Powered Nuclear Regulatory Compliance Checker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parscoders/caelus-compliance",
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
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "isort>=5.0",
        ],
        "web": [
            "streamlit>=1.20.0",
            "streamlit-extras>=0.2.0",
            "plotly>=5.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "caelus=main:main",
            "caelus-test=simple_tester:run_simple_test",
            "caelus-web=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.html", "*.json", "*.txt"],
    },
    zip_safe=False,
) 