#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bumpversion>=0.6.0",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r>=0.2.1",
    "pytest-runner>=5.2",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.4.3",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

step_workflow_requirements = [
    "cloudpickle>=1.5.0",  # needed for distributed
    "boto3==1.15",
    "bokeh>=2.1.0",
    "dask[bag]>=2.21.0",
    "boto3==1.15",  # pin boto3 to resolve CI issues
    "dask_jobqueue>=0.7.0",
    # SPECIAL RELEASE OF DATASTEP TO RESOLVE MALFORMED MANIFEST
    # https://github.com/AllenCellModeling/datastep/tree/special-release
    #
    # Fortunately we left the 0.0.* series of releases empty
    # New datastep projects will by default use the 0.1.* series of releases
    "datastep==0.0.1",
    "distributed>=2.21.0",
    "docutils<0.16",  # needed for botocore (quilt dependency)
    "fire",
    "fsspec>=0.7.4",  # needed for dask[bag]
    "partd>=1.1.0",  # needed for dask[bag]
    "prefect>=0.12.5",
    "python-dateutil<=2.8.0",  # needed for quilt3 (datastep dependency)
    "aics_dask_utils>=0.2.0",  # needed for multiple steps, e.g., DistributedHandler
]

requirements = [
    *step_workflow_requirements,
    # project requires
    "numpy",
    "pandas",
    "Pillow",
    "pyarrow",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "scipy",
    "scikit-image",
    "aicsimageio",
    "aicsimageprocessing",
    "image-classifier-3d",
]


aics_data_requirements = [
    "lkaccess>=1.4.20",
]

integrated_cell_requirements = [
    "torch==1.2.0",
    "torchvision==0.2.1",
    "natsort==5.3.3",
    "pytorch_integrated_cell @ git+https://git@github.com/AllenCellModeling/pytorch_integrated_cell@feature/updated_dependencies#egg=pytorch_integrated_cell",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "aics": aics_data_requirements,
    "all": [*requirements, *dev_requirements, *integrated_cell_requirements],
}

setup(
    author="Allen Institute for Cell Science",
    author_email="jacksonb@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Workflow to manage processing of FOVs and Cells for the Cell Variance Analysis program.",
    entry_points={"console_scripts": ["cvapipe=cvapipe.bin.cli:cli"]},
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="cvapipe",
    name="cvapipe",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="cvapipe/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/aics-int/cvapipe",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.1.0",
    zip_safe=False,
)
