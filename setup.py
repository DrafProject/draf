from pathlib import Path

from setuptools import find_packages, setup

long_description = Path("README.md").read_text().strip()

setup(
    name="draf",
    version="0.1.2",
    author="Markus Fleschutz",
    author_email="mfleschutz@gmail.com",
    description="Demand Response Analysis Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mfleschutz/draf",
    license="LGPLv3",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7,<3.8",
    install_requires=[
        "appdirs",
        "elmada",
        "gsee",
        "gurobipy",
        "holidays",
        "matplotlib",
        "mpu",
        "numpy",
        "pandas",
        "pyomo>=5.7",
        "seaborn",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "black",
            "bump2version",
            "isort",
            "mypy",
            "pytest-cov",
            "pytest-mock",
            "pytest-xdist",
            "pytest",
        ],
        "jupyter": ["jupyter", "jupytext"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "energy systems",
        "optimization",
        "mathematical programming",
        "demand response",
        "energy hubs",
        "distributed energy resources",
    ],
)
