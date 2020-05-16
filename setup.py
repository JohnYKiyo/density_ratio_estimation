from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="densityratio",
    version="0.1.0",
    license="MIT License + LICENSE file",
    description="A Python Package for Direct density estimation by unconstrained Least-Squares Importance Fitting (uLSIF).",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Yu Kiyokawa",
    author_email='dummn.marionette.7surspecies@gmail.com',
    url="https://github.com/JohnYKiyo/density_ratio_estimation",
    keywords='density ratio estimation',
    python_requires=">=3.6.0",
    packages=['densityratio'],
    package_dir = {'densityratio': 'src'},
    install_requires=[
        'jax>=0.1.57',
        'jaxlib>=0.1.37',
        'ipython>=7.12.0'
    ]
)