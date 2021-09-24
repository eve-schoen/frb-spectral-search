import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frb_spectral_search",
    version="0.0.1",
    author="Eve Schoen",
    author_email="eveschn@mit.edu",
    description="Utilities for CHIME/FRB scintillation analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eve-schoen/frb_spectral_search",
    project_urls={
    "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="frb_spectral_search"),
    python_requires=">=3.6",
    )
