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
#pkg_dir_path = os.path.join(root_dir_path, "bqskit")
#readme_path = os.path.join(root_dir_path, "README.md")
#version_path = os.path.join(pkg_dir_path, "version.py")


setup(
      name = "gso",
      packages = ["gso"],
      version = "1.0.2",  
      license="MIT",        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
      description = "Package for finding minimum-edge representatives",
      author = "Hemant Sharma",
      url = "https://github.com/arr0w-hs/graph_state_optimization",
      download_url = "https://github.com/user/reponame/archive/v_01.tar.gz",
      install_requires = ["numpy", "pandas", "matplotlib", "cvxpy"],
      classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Intended Audience :: Quantum engineers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Software Development :: Libraries :: Python Modules",
      ],
)