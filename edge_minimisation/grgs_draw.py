#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 17:13:01 2025

@author: hsharma4
"""

import sys
import os
sys.path.append('..')

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
import matplotlib.pyplot as plt
import networkx as nx
from gso.edm_sa import EDM_SimAnnealing as sa

def rgs_graph(num):
    G = nx.complete_graph(num)
    for i in range(num):
        G.add_node(num+i)
    
        G.add_edge(i, num+i)
    
    return G

def add_leaves(ingraph):
    outgraph = ingraph.copy()
    n = outgraph.number_of_nodes()

    for i in range(n):
        outgraph.add_node(n+i)
        outgraph.add_edge(i, n+i)

    return outgraph

if __name__=="__main__":
    G=nx.Graph()
    edge_list1 = [(0, 10), (0, 13), (0, 14), (0, 16), (0, 17), (1, 9),
                  (1, 11), (1, 12), (1, 13), (1, 14), (1, 16), (1, 17),
                  (2, 10), (2, 13), (2, 14), (2, 16), (2, 17), (3, 10),
                  (3, 12), (3, 13), (3, 14), (3, 15), (4, 9), (4, 10),
                  (4, 11), (4, 12), (5, 9), (5, 10), (5, 11), (5, 12),
                  (6, 12), (6, 15), (6, 16), (6, 17), (7, 9), (7, 10),
                  (7, 11), (7, 15), (7, 16), (7, 17), (8, 9), (8, 10),
                  (8, 11), (8, 15), (8, 16), (8, 17)]
    G.add_edges_from(edge_list1)
    
    
    # G = nx.erdos_renyi_graph(10, p)
    # G = nx.erdos_renyi_graph(5, 0.9999)
    
    # G = nx.complete_graph(6)
    pos = nx.bipartite_layout(G, list(range(9)))
    plt.figure()
    nx.draw_networkx(G, pos=pos, with_labels=False)
    plt.draw()
    # plt.savefig(dir_name + "/no_leaves" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
    plt.show(block=False)
    
    
    
    sa1 = sa(G, 10000, 100)
    gout, _, _ = sa1.simulated_annealing("number of edges")
    
    n = G.number_of_nodes()
    print(n)
    print(G.number_of_edges())
    print(gout.number_of_edges())
    
    # pos = nx.bipartite_layout(gout, list(range(9)))
    plt.figure()
    nx.draw_networkx(gout, with_labels=False)
    plt.draw()
    # plt.savefig(dir_name + "/mer" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
    plt.show(block=False)
    
    
    pos = nx.bipartite_layout(G, list(range(9)))
    # print(pos)
    G = add_leaves(G)
    for v in range(18):
    
        if v < 9:
            pos[18+v] = (pos[v][0]-1, pos[v][1])
        else:
            pos[18+v] = (pos[v][0]+1, pos[v][1])
    print(nx.number_of_edges(G))
    
    # pos = nx.bipartite_layout(G, list(range(18)))
    # print(pos)
    plt.figure()
    nx.draw_networkx(G, pos=pos, with_labels=False)
    plt.draw()
    # plt.savefig(dir_name + "/leaves" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
    plt.show(block=False)
