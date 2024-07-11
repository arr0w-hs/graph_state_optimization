#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:46:52 2024

@author: hsharma4

min edge repre test for adcock's orbits
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import networkx as nx
from edm_sa_ilp import edm_sa_ilp

plt.rcParams.update({'font.size': 12})


data_location = "/orbit.pkl"

with open(dir_name+data_location, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

print(data_dict_loaded.keys())

n = 8

orbit_list = data_dict_loaded[n]
num_orbits = len(orbit_list)
#print(num_orbits, "num_orbits")

input_edges = []
sa_edges = []
ilp_edges = []
y = []
x = []
y1 = []
min_cls = []
max_cls = []

for i, ele in enumerate(orbit_list):
    
    
    
    
    #if i != 5:
    #    continue

    x.append(i)
    num_of_graphs = len(ele)
    y.append(num_of_graphs)
    
    #print(num_of_graphs, "num graphs")
    #for elem in ele:
        #print((elem))
    #for j in range(num_of_graphs):
    #graph_edges = ele[j]

    #random_choice = np.random.randint(1, num_of_graphs)
    #graph_edges = ele[random_choice]
    #print(graph_edges)
    #    G = nx.Graph()
    #    G.add_edges_from(elem)
    #    gin_edges = G.number_of_edges()
        #print(G.number_of_edges())
        
        
    #    plt.figure()
    #    nx.draw_networkx(G)
    #    plt.draw()
    #    plt.show(block = False)
    G_min = nx.Graph()
    G_min.add_edges_from(ele[0])
    y1.append(G_min.number_of_edges())
    #G_max = nx.Graph()
    #G_max.add_edges_from(ele[-1])
    
    #plt.figure()
    #nx.draw_networkx(G_min)
    #plt.draw()
    #plt.show(block = False)
    
    #min_cls.append(nx.average_clustering(G_min))
    #max_cls.append(nx.average_clustering(G_max))
    
    #gmin_edges = G_min.number_of_edges()

fig, ax1 = plt.subplots()
color = 'tab:red'
#fig.suptitle(str(n)+" vertices")
ax1.set_xlabel('orbit number')
ax1.set_ylabel('# graphs in orbit', color=color)
ax1.plot(x, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Edge num in MER', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y1, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.show()

plt.figure()
plt.hist(y1, bins = range(np.min(y1), np.max(y1)+2))
plt.savefig(dir_name + "/hist_of_MERs_with_num_of_edges " + str(n) + ".png",
    dpi=800, format="png", bbox_inches = 'tight')

"""
    output = edm_sa_ilp(G, 100, 100)

    G_out = output[0]
    gout_edges = G_out.number_of_edges()

    sa_edges.append(output[2])
    ilp_edges.append(gout_edges)
    input_edges.append(gin_edges)

    if gout_edges == gmin_edges:
        y.append(1)
    else:
        y.append(0)


    
    
plt.figure()
plt.grid()
plt.plot(x, min_cls)#, x, max_cls)
plt.title("Orbit test for "+str(n)+" vertices")
plt.ylabel('Result')
plt.xlabel('Orbit')
#plt.savefig(dir_name + "/test for vertices " + str(n) + ".png",
#    dpi=800, format="png", bbox_inches = 'tight')
#plt.savefig(dir_name + "/test for vertices " + str(n) + ".svg",
#   dpi=800, format="svg", bbox_inches = 'tight')
    #plt.legend(["Initial edges", "New metric", "Simulated annealing"])
    
    
    
fig, ax1 = plt.subplots()
color = 'tab:red'
#fig.suptitle(str(n)+" vertices")
ax1.set_xlabel('orbit number')
ax1.set_ylabel('# graphs in orbit', color=color)
ax1.plot(x, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Edge num in MER', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y1, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.show()
#fig.savefig(dir_name + "/vertices " + str(n) + ".png",
#    dpi=800, format="png", bbox_inches = 'tight')
#fig.savefig(dir_name + "/vertices " + str(n) + ".svg",
#   dpi=800, format="svg", bbox_inches = 'tight')    
    
    """