#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:52:25 2024

@author: hsharma4

GSO: Graph State Optimisation
Package for finding minimum-edge representatives of LC-equivalent graph states
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("gso")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Public API re-exports
from .base_lc import *
from .edm_ilp import *
from .edm_sa_ilp import *
from .edm_sa import *
from .ILP_minimize_edges import *
from .ILP_SVMinor import *
from .ILP_VMinor import *

# Optional: don’t pull in test helpers by default
# (users can still `import gso.edm_test` if needed)

__all__ = [
    "__version__",
    "edm_sa",
    "edm_ilp",
    "edm_sa_ilp",
]
