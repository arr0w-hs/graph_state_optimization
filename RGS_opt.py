#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:14:44 2024

@author: hsharma4
RGS optimization and comparison with SE
it has edge lists of 4 graphs and a function for creating RGS
uncomment the edge lists to create the graphs
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
#from base_lc import Greedy as greedy
#from base_sa import SimAnnealing as sa

from edm_sa_ilp import edm_sa_ilp
from edm_sa import EDM_SimAnnealing as sa

l1 = []
l2 = []
l3 = []
x = []

#n = 5
#G = nx.complete_graph(n)
#for i in range(n):
#    G.add_node(n+i)
#    G.add_edge(i, n+i)
G = nx.Graph()
"""graph 1  uncomment the lines below to generate the graph"""
#G.add_edges_from([(0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (2, 7), (3, 6), (4, 6), (6, 7)])
"""graph 2"""
#G.add_edges_from([(0, 1), (1, 6), (6, 2), (2, 3), (3, 5), (5, 0), (0, 7), (7, 1), (7, 3), (7, 2), (7, 4), (4, 6), (4, 5)])
"""graph 3"""
#G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 10), (1, 2), (1, 5), (1, 8), (1, 9), (2, 5), (2, 7), (3, 4), (3, 5), (3, 6), (3, 9), (4, 5),
#(4, 7), (4, 10), (6, 7), (6, 8), (6, 9), (7, 10), (8, 9), (9, 10)])
"""MER has 23 edges"""

"""graph 4"""
G.add_edges_from([(0, 2), (0, 6), (0, 8), (0, 10), (0, 14), (0, 16), (1, 3), (1, 4), (1, 6), (1, 9), (1, 10), (1, 13), (1, 15), (1, 16), (2, 4), (2, 5), (2, 7), (2,
10), (2, 11), (2, 14), (2, 16), (3, 8), (3, 9), (3, 10), (3, 11), (3, 16), (4, 6), (4, 7), (4, 9), (4, 12), (4, 13), (4, 16), (5, 6), (5, 7), (5, 8),
(5, 9), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 8), (6, 10), (6, 11), (6, 12), (6, 14), (6, 16), (7, 9), (7, 10), (7, 12), (7, 15), (7,
16), (8, 10), (8, 11), (8, 13), (8, 16), (9, 10), (9, 11), (9, 14), (9, 15), (9, 16), (10, 13), (10, 16), (11, 13), (11, 14), (11, 16), (12, 14),
(12, 16), (13, 15), (13, 16), (14, 16), (15, 16)])
#G = nx.grid_graph(dim=(2,4))
#print(G.number_of_edges())

#plt.figure()
#nx.draw_networkx(G)
#plt.draw()


def rgs_graph(num):
    G = nx.complete_graph(n)
    for i in range(n):
        G.add_node(n+i)
    
        G.add_edge(i, n+i)
    
    return G



for i in range(1):
    #n = 7
    #print(i)
    #x.append(i+1)
    
    
    #G = rgs_graph(n)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    plt.figure()
    nx.draw_networkx(G)
    plt.draw()
    
    
    l1.append(G.number_of_edges())
    
    #opt = greedy(G)
    #edges, opg = opt.greedy_minimisation()
    
    #l2.append(edges)
    
    output = edm_sa_ilp(G, 100, 100)
    #sa1 = sa(G, 100, 100)
    #g_out, y_list, ui_list = sa1.simulated_annealing("number of edges")
    
    #print(g_out.number_of_edges())
    l3.append(output[3])
    print(output[3])
    #print(g_out.number_of_edges())
    plt.figure()
    #nx.draw_networkx(g_out)
    nx.draw_networkx(output[0])
    plt.draw()
    print("-----", str(output[4]), "seconds -----")
#x = np.asarray(x)
#print(g_out.number_of_edges())
#plt.figure()
#plt.grid()
#plt.plot(x, l1, '-')
#plt.plot(x,x)
#plt.plot(x, l2, 'o')
#plt.plot(x, l3, '-')

#print(g_out.edges())