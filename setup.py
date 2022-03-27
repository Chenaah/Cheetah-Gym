import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "dog",
    version = "1.0",
    author = "Chen",
    author_email = "",
    description = (""),
    keywords = "Reinforcement Learning, Robot Learning",
    packages=['dog'],
    long_description=read('README.md'),
)