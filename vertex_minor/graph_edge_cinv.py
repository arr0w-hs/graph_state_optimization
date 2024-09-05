#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:22:38 2024

@author: hsharma4
convert networkx graph into 
"""


import sys
import os
from pathlib import Path
#dir_name = os.path.dirname(__file__)
#os.chdir(dir_name)
sys.path.append('..')



import time as time
import matplotlib.pyplot as plt
import networkx as nx
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
#from base_lc import Greedy as greedy
#from base_sa import SimAnnealing as sa

#from optimizer.edm_ilp import ilp_minimize_edges
from optimizer.ILP_VMinor import has_VM
from optimizer.edm_sa import EDM_SimAnnealing as sa
from optimizer.ILP_minimize_edges import minimize_edges as ilp_minimize_edges

G = nx.erdos_renyi_graph(4, 0.8)
H = nx.star_graph(4)

plt.figure()
nx.draw_networkx(H)#, pos=pos)
plt.show()
plt.figure()


print(nx.adjacency_data(H))
#print(H.nodes.data())
print(range(10))

