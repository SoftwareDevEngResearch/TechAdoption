''' Erin Peiffer, 16 April 2020
	Setup.py
'''
from codecs import open
from os import path
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
	

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'TechAdoption', '_version.py')) as version_file:
    exec(version_file.read())

with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

#with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
#    changelog = changelog_file.read()

desc = readme + '\n\n' + changelog
try:
    import pypandoc
    long_description = pypandoc.convert_text(desc, 'rst', format='md')
    with open(path.join(here, 'README.rst'), 'w') as rst_readme:
        rst_readme.write(long_description)
except (ImportError, OSError, IOError):
    long_description = desc

''' update install requires and tests '''
install_requires = [
    'numpy',
]

tests_require = [
    'pytest',
    'pytest-cov',
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name='TechAdoption',
    version='0.1.0',
    description='Identify top factors that predict rates of adoption',
	long_description=open('README.txt').read()
    author='Erin Peiffer',
    author_email='peiffer.eringmail.com',
    url='https://github.com/Epeiffer1',
    classifiers=[
        'License :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research/Development Practitioners',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    license='MIT-Clause',
    python_requires='>=3',
    zip_safe=False,
    packages=['TechAdoption', 'TechAdoption.tests'],
    include_package_data=True,

    # or you can specify explicitly:
    package_data={
        'TechAdoption': ['assets/*.txt'] #add other data files like csv or text files
        },
)