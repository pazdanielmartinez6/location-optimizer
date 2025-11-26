"""Setup script for ATM Location Optimizer package."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="atm-location-optimizer",
    version="1.0.0",
    author="Daniel Paz Martinez",
    description="Optimize ATM placement using real-world travel times and advanced algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/atm-location-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "geopandas>=0.10.0",
        "folium>=0.12.0",
        "shapely>=1.8.0",
        "pyproj>=3.2.0",
        "osmnx>=1.1.0",
        "contextily>=1.2.0",
        "requests>=2.26.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "atm-optimizer=atm_optimizer.cli:main",
        ],
    },
)
