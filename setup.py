#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="John T. Foster",
    author_email='johntfosterjr@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="1,2, and 3D finite difference solver that supports n-dof and nonlinear parallel solves with PyTrilinos",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='PyTriFD',
    name='PyTriFD',
    packages=find_packages(include=['PyTriFD', 'PyTriFD.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/johntfoster/PyTriFD',
    version='0.1.0',
    zip_safe=False,
)
