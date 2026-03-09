from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="bessel_seg",
    version="0.1.0",
    description=(
        "Automated neuron segmentation pipeline for Bessel beam "
        "two-photon fluorescence microscopy images"
    ),
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    extras_require={
        "optional": [
            "torch>=1.12",
            "napari>=0.4.17",
            "cellpose>=2.0",
            "suite2p>=0.13",
        ]
    },
    entry_points={
        "console_scripts": [
            "bessel-run=bessel_seg.pipeline:run_pipeline",
        ]
    },
)
