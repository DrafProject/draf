from pathlib import Path

from setuptools import find_packages, setup

exec(Path("draf/_version.py").read_text().strip())  # Set the __version__ variable

long_description = Path("README.md").read_text().strip()

setup(
    name="draf",
    version=__version__,
    author="Markus Fleschutz",
    author_email="mfleschutz@gmail.com",
    description="Demand Response Analysis Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mfleschutz/draf",
    license="LGPLv3",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=["numpy", "holidays", "pandas", "matplotlib", "seaborn", "pyomo>=5.7"],
    extras_require={"dev": ["pytest", "pytest-cov", "pytest-mock", "mypy"]},
    classifiers=[
        "Development Status :: 4 - Alpha",
        "Environment :: Console",
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
        "microgrids",
        "distributed energy resources",
    ],
)
