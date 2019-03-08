#!/usr/bin/env python
from setuptools import setup

sdk_package_name = 'alooma_pysdk'
packages = [sdk_package_name]


setup(
    name=sdk_package_name,
    packages=packages,
    package_data={sdk_package_name: ['alooma_ca']},
    version='3.0.0',
    description='An easy-to-integrate SDK for your Python apps to report '
                'events to Alooma',
    url='https://github.com/Aloomaio/python-sdk',
    author='Alooma',
    author_email='integrations@alooma.com',
    keywords=['python', 'sdk', 'alooma', 'pysdk'],
    install_requires=open("requirements.txt").readlines(),
    tests_require=open("requirements-tests.txt").readlines()
)
