"""GNF Evidence Engine — Continuous meta-analysis + TSA + provenance engine."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="gnf-evidence-engine",
    version="0.1.0",
    author="Global Neuro Foundry",
    author_email="science@globalneuro.org",
    description=(
        "Continuous meta-analysis, Trial Sequential Analysis, and "
        "provenance tracking for translational neuroscience research"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GlobalNeuroFoundry/gnf-evidence-engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "mypy>=1.5.0",
            "ruff>=0.0.292",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "gnf-dashboard=dashboard.app:main",
        ]
    },
)
