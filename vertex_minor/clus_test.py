#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:36:00 2024

@author: hsharma4
"""

import sys
import os
from pathlib import Path
dir_name = os.path.dirname(__file__)
os.chdir(dir_name)
sys.path.append('..')

import cvxpy as cvx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import time

from optimizer.edm_sa import EDM_SimAnnealing as sa

#n = 5
#G = nx.complete_graph(n)
#for i in range(n):
#    G.add_node(n+i)
#    G.add_edge(i, n+i)
G = nx.Graph()
"""graph 1  uncomment the lines below to generate the graph"""
G.add_edges_from([(0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (2, 7), (3, 6), (4, 6), (6, 7)])
"""graph 2"""
#G.add_edges_from([(0, 1), (1, 6), (6, 2), (2, 3), (3, 5), (5, 0), (0, 7), (7, 1), (7, 3), (7, 2), (7, 4), (4, 6), (4, 5)])
"""graph 3"""
#G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 10), (1, 2), (1, 5), (1, 8), (1, 9), (2, 5), (2, 7), (3, 4), (3, 5), (3, 6), (3, 9), (4, 5),
#(4, 7), (4, 10), (6, 7), (6, 8), (6, 9), (7, 10), (8, 9), (9, 10)])
"""MER has 23 edges"""

"""graph 4"""
#G.add_edges_from([(0, 2), (0, 6), (0, 8), (0, 10), (0, 14), (0, 16), (1, 3), (1, 4), (1, 6), (1, 9), (1, 10), (1, 13), (1, 15), (1, 16), (2, 4), (2, 5), (2, 7), (2,
#10), (2, 11), (2, 14), (2, 16), (3, 8), (3, 9), (3, 10), (3, 11), (3, 16), (4, 6), (4, 7), (4, 9), (4, 12), (4, 13), (4, 16), (5, 6), (5, 7), (5, 8),
#(5, 9), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 8), (6, 10), (6, 11), (6, 12), (6, 14), (6, 16), (7, 9), (7, 10), (7, 12), (7, 15), (7,
#16), (8, 10), (8, 11), (8, 13), (8, 16), (9, 10), (9, 11), (9, 14), (9, 15), (9, 16), (10, 13), (10, 16), (11, 13), (11, 14), (11, 16), (12, 14),
#(12, 16), (13, 15), (13, 16), (14, 16), (15, 16)])
#G = nx.grid_graph(dim=(2,4))
#print(G.number_of_edges())
#n1 = 4
#n2 = 4
#G = nx.complete_bipartite_graph(n1,n2)
#plt.figure()
#nx.draw_networkx(G)
#plt.draw()


def rgs_graph(num):
    G = nx.complete_graph(num)
    for i in range(num):
        G.add_node(num+i)
    
        G.add_edge(i, num+i)
    
    return G

time_list = []
g_list = []
o_list = []
for i in range(100):
    #print(i)
    G = nx.erdos_renyi_graph(50, 0.6)
    g_list.append(G.number_of_edges())
    sa1 = sa(G, 1000, 1000)
    t1=time.time()
    g_out = sa1.simulated_annealing("number of edges")
    time_list.append(time.time()-t1)
    #print(time.time()-t1)
    o_list.append(g_out[0].number_of_edges())

    #nx.draw_networkx(G)#, pos=pos)
    #plt.show
    #plt.figure()
    # nx.draw_networkx(g_out)#, pos=pos)
    # list1 = list(nx.clustering(G).values())
    # list2 = list(nx.clustering(g_out).values())
    # c = [a + b for a, b in zip(list1, list2)]
    # print(list1)
    # print(list2)
    # print(c)
print(np.average(time_list))
print(np.average(g_list))
print(np.average(o_list))
    #plt.show()

    
    #print(g_out.number_of_edges())
    #l3.append(output[3])
    #print(output[3])
    #print(g_out.number_of_edges())
    #plt.figure()
    #nx.draw_networkx(g_out)
    #nx.draw_networkx(output[0])#, pos = pos)
    #plt.draw()
    
    #plt.figure()
    #nx.draw_networkx(output1[0])
    #plt.draw()
    #print("-----", str(output[4]), "seconds -----")
