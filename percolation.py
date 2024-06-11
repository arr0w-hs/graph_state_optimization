#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:07:35 2024

@author: hsharma4
graph percolation
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

#n = 10
##G = nx.complete_graph(n)
#for i in range(n):
#    G.add_node(n+i)
#    G.add_edge(i, n+i)
G = nx.Graph()
#G.add_edges_from([(0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (2, 7), (3, 6), (4, 6), (6, 7)])
G.add_edges_from([(0, 1), (1, 6), (6, 2), (2, 3), (3, 5), (5, 0), (0, 7), (7, 1), (7, 3), (7, 2), (7, 4), (4, 6), (4, 5)])
#G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 10), (1, 2), (1, 5), (1, 8), (1, 9), (2, 5), (2, 7), (3, 4), (3, 5), (3, 6), (3, 9), (4, 5),
#(4, 7), (4, 10), (6, 7), (6, 8), (6, 9), (7, 10), (8, 9), (9, 10)])
#G.add_edges_from([(0, 2), (0, 6), (0, 8), (0, 10), (0, 14), (0, 16), (1, 3), (1, 4), (1, 6), (1, 9), (1, 10), (1, 13), (1, 15), (1, 16), (2, 4), (2, 5), (2, 7), (2,
#10), (2, 11), (2, 14), (2, 16), (3, 8), (3, 9), (3, 10), (3, 11), (3, 16), (4, 6), (4, 7), (4, 9), (4, 12), (4, 13), (4, 16), (5, 6), (5, 7), (5, 8),
#(5, 9), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 8), (6, 10), (6, 11), (6, 12), (6, 14), (6, 16), (7, 9), (7, 10), (7, 12), (7, 15), (7,
#16), (8, 10), (8, 11), (8, 13), (8, 16), (9, 10), (9, 11), (9, 14), (9, 15), (9, 16), (10, 13), (10, 16), (11, 13), (11, 14), (11, 16), (12, 14),
#(12, 16), (13, 15), (13, 16), (14, 16), (15, 16)])
#G = nx.grid_graph(dim=(2,4))
plt.figure()
nx.draw_networkx(G)
plt.draw()
#print((G.number_of_edges()))
#G.remove_node(2)
for i in range(G.number_of_nodes()):
    #n = 100
    #print(i)
    k_max = 500
    G = nx.Graph()
    #G.add_edges_from([(0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (2, 7), (3, 6), (4, 6), (6, 7)])
    G.add_edges_from([(0, 1), (1, 6), (6, 2), (2, 3), (3, 5), (5, 0), (0, 7), (7, 1), (7, 3), (7, 2), (7, 4), (4, 6), (4, 5)])
    #G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 10), (1, 2), (1, 5), (1, 8), (1, 9), (2, 5), (2, 7), (3, 4), (3, 5), (3, 6), (3, 9), (4, 5),
    #(4, 7), (4, 10), (6, 7), (6, 8), (6, 9), (7, 10), (8, 9), (9, 10)])
    x.append(i)
    G.remove_node(i)
    #print(list(G.edges()))
    #G = nx.complete_graph(n)
    #for i in range(n):
    #    G.add_node(n+i)
    #    G.add_edge(i, n+i)
    
    plt.figure()
    nx.draw_networkx(G)
    plt.draw()
           
    #print((G.number_of_edges()))
    l1.append(G.number_of_edges())
    
    opt = greedy(G)
    edges, opg = opt.greedy_minimisation()
    
    l2.append(edges)
    sa1 = sa(G, k_max, 100)
    g_out, y_list, ui_list = sa1.simulated_annealing("number of edges")
    l3.append(g_out.number_of_edges())
    
    #plt.figure()
    #nx.draw_networkx(g_out)
    #plt.draw()
    
x = np.asarray(x)
#print(g_out.degree())
plt.figure()
plt.grid()
plt.plot(x, l1, '-o')
#plt.plot(x,x)
#plt.plot(x, l2, '-o')
plt.plot(x, l3, '-o')
