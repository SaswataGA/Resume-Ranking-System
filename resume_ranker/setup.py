"""
Setup configuration for the Resume Ranking System.
Allows the package to be installed with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
req_path = Path(__file__).parent / "requirements.txt"
if req_path.exists():
    requirements = [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="resume-ranking-system",
    version="1.0.0",
    author="Resume Ranking System Authors",
    description="Skill-Based Resume Ranking System with Explainable AI",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/resume-ranking-system",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "resume-rank=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    include_package_data=True,
    package_data={
        "skill_extraction": ["skill_ontology.json"],
        "data": ["sample/*.txt", "sample/*.json"],
    },
)
