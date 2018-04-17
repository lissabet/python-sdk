#!/usr/bin/env python
from setuptools import setup, find_packages
try:
    from pip.req import parse_requirements
except ImportError:
    from pip._internal.req import parse_requirements

install_reqs=parse_requirements("requirements.txt", session=False)
reqs = [str(ir.req) for ir in install_reqs]
sdk_package_name = 'alooma_pysdk'
packages=[sdk_package_name]


setup(
    name=sdk_package_name,
    packages=packages,
    package_data={sdk_package_name: ['alooma_ca']},
    version='2.0.2',
    description='An easy-to-integrate SDK for your Python apps to report '
                'events to Alooma',
    url='https://github.com/Aloomaio/python-sdk',
    author='Alooma',
    author_email='integrations@alooma.com',
    keywords=['python', 'sdk', 'alooma', 'pysdk'],
    install_requires=install_reqs
)
