from setuptools import setup, find_packages  # type: ignore

setup(
    name="preprocessing-pipeline",
    version="0.1.3",
    description="Preprocessing pipeline for neural feature extraction",
    author="Kevin Davis",
    author_email="kevin.davis@miami.edu",
    packages=find_packages(exclude=("tests")),
    include_package_data=True,
)
