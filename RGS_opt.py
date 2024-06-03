#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:14:44 2024

@author: hsharma4
RGS optimization and comparison with SE
"""
from pathlib import Path

import sys
import os


import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
from base_lc import Greedy as greedy
from base_sa import SimAnnealing as sa

l1 = []
l2 = []
l3 = []
x = []


for i in range(20):
    n = i+1
    #print(i)
    x.append(i+1)
    
    G = nx.complete_graph(n)
    for i in range(n):
        G.add_node(n+i)
        G.add_edge(i, n+i)
    
    
    
    #plt.figure()
    #nx.draw_networkx(G)
    #plt.draw()
    
    
    l1.append(G.number_of_edges())
    
    opt = greedy(G)
    edges, opg = opt.greedy_minimisation()
    
    l2.append(edges)
    sa1 = sa(G, 100, 100)
    g_out, y_list, ui_list = sa1.simulated_annealing("number of edges")
    l3.append(g_out.number_of_edges())

x = np.asarray(x)

plt.figure()
plt.grid()
plt.plot(x, l1)
#plt.plot(x, l2)
plt.plot(x, l3)
