#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:26:51 2024

@author: hsharma4
"""

import sys
import os

import pickle

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams.update({'font.size': 12})

from local_complementation import *


G = nx.fast_gnp_random_graph(8, 0.7)
G_in = G
plt.figure()
nx.draw_networkx(G)
plt.draw()


ff = nx.laplacian_spectrum(G)
plt.figure()
plt.hist(ff)

#ff = nx.adjacency_spectrum(G)
#plt.figure()
#plt.hist(ff)


print(nx.algebraic_connectivity(G), "intial ac")

cl_list = []
num_edges_list = []
vert_list = []

x=[]
flag = 1
i = 0
while flag == 1:
    num_edges_list.append(G.number_of_edges())
    cl_list.append(nx.average_clustering(G))
    G, vert, flag = apply_lc_clustering(G)
    vert_list.append(vert)
    i += 1
    flag = 1
    if i >= 10:
        flag = 0
print(nx.algebraic_connectivity(G), "lc ac")
x = np.linspace(1, i, i)
plt.figure()
nx.draw_networkx(G)
plt.draw()

ff = nx.laplacian_spectrum(G)
plt.figure()
plt.hist(ff)

plt.figure()
plt.grid()
plt.plot(x, num_edges_list)#, s = 5, c = "blue")
plt.ylabel('Number of edges')
plt.xlabel("Iteration")


#plt.figure()
#plt.grid()
#plt.plot(x, vert_list)#, s = 5, c = "blue")
#plt.ylabel('vert_list')
#print(nx.average_clustering(G), "clustering")


G = G_in
cl_list = []
num_edges_list = []
vert_list = []

x = []
i=0
flag = 1
while flag == 1:
    num_edges_list.append(G.number_of_edges())
    cl_list.append(nx.average_clustering(G))
    G, vert, flag = apply_new_metric(G)
    vert_list.append(vert)
    flag = 1
    i += 1
    if i >= 10:
        flag = 0

print(nx.algebraic_connectivity(G), "nm ac")
plt.figure()
nx.draw_networkx(G)
plt.draw()

ff = nx.laplacian_spectrum(G)
plt.figure()
plt.hist(ff)

x = np.linspace(1, i, i)
plt.figure()
plt.grid()
plt.plot(x, num_edges_list)#, s = 5, c = "blue")

#plt.figure()
#plt.grid()
#plt.plot(x, vert_list)#, s = 5, c = "blue")
#plt.ylabel('vert_list')
#print(nx.average_clustering(G), "new metric")

"""
cl_list = []
num_edges_list = []
vert_list = []
G = G_in
x = []
i=0
flag = 1
while flag == 1:
    num_edges_list.append(G.number_of_edges())
    cl_list.append(nx.average_clustering(G))
    G, vert, flag = apply_lc_triangles(G)
    vert_list.append(vert)
    i += 1
    #print(i)
    flag = 1
    if i >= 10:
        flag = 0

print(nx.algebraic_connectivity(G), "tr ac")
plt.figure()
nx.draw_networkx(G)
plt.draw()
x = np.linspace(1, i, i)
plt.figure()
plt.grid()
plt.plot(x, num_edges_list)#, s = 5, c = "blue")
plt.ylabel('Number of edges')
plt.xlabel("Iteration")
#print(nx.average_clustering(G), "triangles")
#plt.figure()
#plt.grid()
#plt.plot(x, vert_list)#, s = 5, c = "blue")
#plt.ylabel('vert_list')
"""