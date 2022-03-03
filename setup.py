from setuptools import find_packages, setup

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seiz_eeg",
    packages=find_packages(),
    version="0.1.2a0",
    description="Data loading and preprocessing of EEG scans for seizure-related ML tasks",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="William Cappelletti",
    author_email="william.cappelletti@epfl.ch",
    license="BSD-3",
    url="https://github.com/WilliamCappelletti/seizure_eeg",
    project_urls={
        "Bug Tracker": "https://github.com/WilliamCappelletti/seizure_eeg/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "pandera",
        "pyEDFlib==0.1.19",
        "scipy",
        "omegaconf",
        "pexpect",
        "tqdm",
    ],
)
