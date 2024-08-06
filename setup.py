#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:52:43 2024

@author: hsharma4
"""


import os
import sys

#sys.path.append(os.path.dirname(__file__))
#dir_name = os.path.dirname(__file__)

from setuptools import setup
from setuptools import find_packages

root_dir_path = os.path.abspath(os.path.dirname(__file__))
#pkg_dir_path = os.path.join(root_dir_path, 'bqskit')
#readme_path = os.path.join(root_dir_path, 'README.md')
#version_path = os.path.join(pkg_dir_path, 'version.py')


setup(
      name = 'gso',
      packages = find_packages(),
      install_requires = ["numpy", "pandas", "matplotlib"])

#from gso.edm import bd_mer
#from .utils import *
#from .vetex_minor import *
