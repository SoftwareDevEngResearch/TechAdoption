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

#with open(path.join(here, 'TechAdoption', '__version.py')) as version_file:
 #   exec(version_file.read())

with open("README.md", "r") as fh:
    long_description = fh.read()


with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

desc = long_description + '\n\n' + changelog
try:
    import pypandoc
    long_description = pypandoc.convert_text(desc, 'rst', format='md')
    with open(path.join(here, 'README.md'), 'w') as rst_readme:
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
	long_description=long_description,
    long_description_content_type="text/markdown",
    author='Erin Peiffer',
    author_email='peiffer.erin@gmail.com',
    url='https://github.com/Epeiffer1',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    license='MIT-Clause',
    python_requires='>=3',
    zip_safe=False,
    packages=['TechAdoption', 'TechAdoption.Test'],
    include_package_data=True,

    # or you can specify explicitly:
    package_data={
        'TechAdoption': ['assets/*.txt'] #add other data files like csv or text files
        },
)