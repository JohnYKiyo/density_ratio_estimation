from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


REQUIRES_PYTHON = '>=3.6.0'

def _requires_from_file(filename):
    return open(filename).read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="densityratio",
    version="0.1.0",
    license="MIT License",
    description="A Python Package for Direct density estimation by unconstrained Least-Squares Importance Fitting (uLSIF).",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Yu Kiyokawa",
    url="https://github.com/JohnYKiyo/density_ratio_estimation",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt')
    #setup_requires=["pytest-runner"],
    #tests_require=["pytest", "pytest-cov"]
)