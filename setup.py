from itertools import chain
from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

extras = {
    "thinplatesplines": ["morphops", "scipy"],
    "movingleastsquares": ["molesq"],
}
extras["all"] = list(set(chain.from_iterable(extras.values())))
extras["test"] = ["pytest"]

setup(
    name="transformnd",
    url="https://github.com/clbarnes/transformnd",
    author="Chris L. Barnes",
    description="ND coordinate transformations",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["transformnd*"]),
    install_requires=["numpy>=1.20", "networkx"],
    extras_require=extras,
    tests_require=["pytest"],
    python_requires=">=3.7, <4.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
