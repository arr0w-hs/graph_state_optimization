#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:14:51 2024

@author: hsharma4
tests for VM runtimes
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


l1 = []
l2 = []
l3 = []
x = []

#n = 5
#G = nx.complete_graph(n)
#for i in range(n):
#    G.add_node(n+i)
#    G.add_edge(i, n+i)
tl = []
tl1 = []
for j in range(11):
    print(j)
    t_list = []
    t_list1 = []
    n = 7+j
    x.append(n)
    for _ in range(5):
        #print(j)
        G = nx.erdos_renyi_graph(n, 0.8)
        H = nx.star_graph(4)
    
        # plt.figure()
        # nx.draw_networkx(G)#, pos=pos)
        # plt.show()
        # plt.figure()
        # nx.draw_networkx(H)#, pos=pos)
        # plt.show()
        #t1 = time.time()
        #sa1 = sa(G, 100, 100)
        #G, y_list, ui_list = sa1.simulated_annealing("number of edges")
        #feasible, G_output = has_VM(G, H, check_LC=True)
        #t_list1.append(time.time()-t1)

        t1 = time.time()
        sa1 = sa(G, 100, 100)
        G, y_list, ui_list = sa1.simulated_annealing("number of edges")
        feasible, G_output = has_VM(G, H, check_LC=True)
        t_list.append(time.time()-t1)
        #else:
            #print("not feasible")

    #if len(t_list) != 0:
    t_avg = (sum(t_list)/len(t_list))
    print(t_avg)
    tl.append(t_avg)

    #t_avg1 = (sum(t_list1)/len(t_list1))
#    print(t_avg)
    #tl1.append(t_avg1)
#print(tl)
plt.figure()
plt.plot(x, tl)
#plt.plot(x, tl1)
plt.ylabel("Runtime (s)")
plt.xlabel("Num vertices")
plt.yscale("log")
plt.grid()
plt.legend(["SA+has_VM", "has_VM"])
#plt.savefig("vm11.png", dpi=1000, format="png", bbox_inches = 'tight')
