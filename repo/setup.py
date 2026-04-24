from setuptools import setup, find_packages

setup(
    name="algebraic-transplant",
    version="1.0.0",
    description=(
        "Algebraic Transplant of Meshless Discrete Operators into Graph Neural "
        "Architectures for Numerically Consistent Neural Operators"
    ),
    author="Amirkeivan Shafiei, Seyed Mojtaba Mosavi Nezhad",
    author_email="k.shafiei@birjand.ac.ir",
    license="MIT",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black", "isort"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
