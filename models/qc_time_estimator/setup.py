import io
import os
import sys
from pathlib import Path
import pip
from setuptools import find_packages, setup


# Package meta-data.
NAME = 'qc_time_estimator'
DESCRIPTION = 'Train and deploy QC time estimator models'
URL = 'https://github.com/MolSSI/ml_models_deploy'
EMAIL = 'doaa.altarawy@gmail.com'
AUTHOR = 'Doaa Altarawy'
REQUIRES_PYTHON = '>=3.6.5'


try:
    if pip.__version__ >= "19.3":
        from pip._internal.req import parse_requirements
        from pip._internal.network.session import PipSession
    elif pip.__version__ >= "10.0" and pip.__version__ < "19.3":
        from pip._internal.req import parse_requirements
        from pip._internal.download import PipSession
    else:  # pip < 10 is not supported
        raise Exception('Please upgrade pip: pip install --upgrade pip')
except ImportError as err:  # for future changes in pip
    print('New breaking changes in pip!!', err)
    sys.exit()


def read_requirements():
    """parses requirements from requirements.txt"""

    install_reqs = parse_requirements('requirements.txt', session=PipSession())
    return [ir.name for ir in install_reqs]


here = os.path.abspath(os.path.dirname(__file__))

# 'README.md' must be in MANIFEST.in
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'qc_time_estimator': ['VERSION']},
    install_requires=read_requirements(),
    extras_require={},
    include_package_data=True,
    license='BSD-3C',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
