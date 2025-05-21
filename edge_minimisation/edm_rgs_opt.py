#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:14:44 2024

@author: hsharma4
RGS optimization and comparison with SE
it has edge lists of 4 graphs and a function for creating RGS
uncomment the edge lists to create the graphs
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
from optimizer.edm_sa_ilp import edm_sa_ilp
from optimizer.edm_sa import EDM_SimAnnealing as sa
from optimizer.ILP_minimize_edges import minimize_edges as ilp_minimize_edges
from optimizer.gsc.is_lc_equiv import are_lc_equiv


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
G2 = nx.Graph()
"""graph 1  uncomment the lines below to generate the graph"""
#G.add_edges_from([(0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (2, 7), (3, 6), (4, 6), (6, 7)])
"""graph 2"""
#G.add_edges_from([(0, 1), (1, 6), (6, 2), (2, 3), (3, 5), (5, 0), (0, 7), (7, 1), (7, 3), (7, 2), (7, 4), (4, 6), (4, 5)])
"""graph 3"""
# G2.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8),
#                    (0, 10), (1, 2), (1, 5), (1, 8), (1, 9), (2, 5), (2, 7),
#                    (3, 4), (3, 5),(3, 6), (3, 9), (4, 5), (4, 7), (4, 10),
#                    (6, 7), (6, 8), (6, 9), (7, 10), (8, 9), (9, 10)])
"""MER has 23 edges given by:"""
# G.add_edges_from([(0, 1), (0, 2), (0, 5), (0, 7), (1, 2), (1, 4), (1, 8), (2, 9),
#  (2, 10), (3, 4), (3, 6), (3, 8), (3, 9), (4, 5), (4, 8), (5, 6),
#  (5, 7), (5, 10), (6, 7), (6, 10), (7, 8), (8, 9), (9, 10)])

"""graph 4"""
# G.add_edges_from([(0, 2), (0, 6), (0, 8), (0, 10), (0, 14), (0, 16), (1, 3), (1, 4), (1, 6), (1, 9), (1, 10), (1, 13), (1, 15), (1, 16), (2, 4), (2, 5), (2, 7), (2,
# 10), (2, 11), (2, 14), (2, 16), (3, 8), (3, 9), (3, 10), (3, 11), (3, 16), (4, 6), (4, 7), (4, 9), (4, 12), (4, 13), (4, 16), (5, 6), (5, 7), (5, 8),
# (5, 9), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 8), (6, 10), (6, 11), (6, 12), (6, 14), (6, 16), (7, 9), (7, 10), (7, 12), (7, 15), (7,
# 16), (8, 10), (8, 11), (8, 13), (8, 16), (9, 10), (9, 11), (9, 14), (9, 15), (9, 16), (10, 13), (10, 16), (11, 13), (11, 14), (11, 16), (12, 14),
# (12, 16), (13, 15), (13, 16), (14, 16), (15, 16)])

"""approximate mer has 51 edges"""
# G.add_edges_from([(0, 1), (0, 5), (0, 10), (0, 12), (0, 15), (0, 16), (1, 2),
#                  (1, 7), (1, 11), (1, 12), (1, 13), (1, 15), (2, 3), (2, 7),
#                  (2, 8), (2, 11), (2, 14), (2, 15), (2, 16), (3, 4), (3, 8),
#                  (3, 9), (3, 10), (3, 14), (3, 15), (4, 6), (4, 9), (4, 10),
#                  (4, 12), (4, 13), (4, 14), (4, 16), (5, 6), (5, 8), (5, 9),
#                  (5, 11), (6, 7), (6, 12), (6, 14), (6, 16), (7, 8), (7, 12),
#                  (8, 10), (9, 13), (9, 15), (9, 16), (10, 12), (11, 13),
#                  (11, 14), (12, 13), (12, 16)])

edge_list1 = [(1, 3), (1, 6), (1, 9), (1, 10), (1, 15), (1, 25), (1, 26), (1, 31),
 (1, 49), (1, 50), (1, 55), (6, 5), (6, 2), (5, 3), (5, 4), (7, 4),
 (7, 3), (7, 9), (7, 10), (7, 15), (7, 25), (7, 26), (7, 31), (7, 49),
 (7, 50), (7, 55), (4, 2), (2, 9), (2, 10), (2, 15), (2, 25), (2, 26),
 (2, 31), (2, 49), (2, 50), (2, 55), (9, 11), (9, 14), (9, 17),
 (9, 18), (9, 23), (9, 57), (9, 58), (9, 63), (14, 13), (14, 10),
 (13, 11), (13, 12), (15, 12), (15, 11), (15, 17), (15, 18), (15, 23),
 (15, 57), (15, 58), (15, 63), (12, 10), (10, 17), (10, 18), (10, 23),
 (10, 57), (10, 58), (10, 63), (17, 19), (17, 22), (17, 25), (17, 26),
 (17, 31), (17, 33), (17, 34), (17, 39), (22, 21), (22, 18), (21, 19),
 (21, 20), (23, 20), (23, 19), (23, 25), (23, 26), (23, 31), (23, 33),
 (23, 34), (23, 39), (20, 18), (18, 25), (18, 26), (18, 31), (18, 33),
 (18, 34), (18, 39), (25, 27), (25, 30), (25, 41), (25, 42), (25, 47),
 (30, 29), (30, 26), (29, 27), (29, 28), (31, 28), (31, 27), (31, 41),
 (31, 42), (31, 47), (28, 26), (26, 41), (26, 42), (26, 47), (33, 35),
 (33, 38), (33, 41), (33, 42), (33, 47), (33, 57), (33, 58), (33, 63),
 (38, 37), (38, 34), (37, 35), (37, 36), (39, 36), (39, 35), (39, 41),
 (39, 42), (39, 47), (39, 57), (39, 58), (39, 63), (36, 34), (34, 41),
 (34, 42), (34, 47), (34, 57), (34, 58), (34, 63), (41, 43), (41, 46),
 (41, 49), (41, 50), (41, 55), (46, 45), (46, 42), (45, 43), (45, 44),
 (47, 44), (47, 43), (47, 49), (47, 50), (47, 55), (44, 42), (42, 49),
 (42, 50), (42, 55), (49, 51), (49, 54), (49, 57), (49, 58), (49, 63),
 (54, 53), (54, 50), (53, 51), (53, 52), (55, 52), (55, 51), (55, 57),
 (55, 58), (55, 63), (52, 50), (50, 57), (50, 58), (50, 63), (57, 59),
 (57, 62), (62, 61), (62, 58), (61, 59), (61, 60), (63, 60), (63, 59),
 (60, 58)]

# edge_list = []
# for u, v in edge_list1:
#     #for v in list(val):
#     print(u,v)
#     edge_list.append((u-1, v-1))

G.add_edges_from(edge_list1)
G = nx.convert_node_labels_to_integers(G)
print(G.number_of_nodes())

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



for i in range(1):
    #n = 7
    #print(i)
    #x.append(i+1)
    #n = G.number_of_nodes()
    #n = n1+n2
    #for i in range(n):
    #    G.add_node(n+i)
    #    G.add_edge(i, n+i)
    #
    #G = rgs_graph(n)
    #print(G.number_of_nodes())
    
    # pos = nx.spring_layout(G)#, list(range(n)))
    
    #G = rgs_graph(6)
    
    print(G.number_of_edges())

    plt.figure()
    nx.draw_networkx(G)
    plt.draw()
    plt.show(block=False)


    # degg = [v for u,v in G.degree()]
    # print(degg)
    


    #opt = greedy(G)
    #edges, opg = opt.greedy_minimisation()
    
    #l2.append(edges)
    #time1 = time.time()
    #output1  = ilp_minimize_edges(G)
    #print(time.time()-time1)
    #print(output1[1])
    #output = edm_sa_ilp(G, 100, 100)
    # output = edm_sa_ilp(G, 1000, 100)
    # print(G.number_of_edges())

    # gout = (output[0])
    # # sa_list.append(output[2])
    # # sa_ilp_list.append(output[3])
    # # runtime_sailp_list.append(output[4])

    # print(gout.edges())
    # print(gout.number_of_edges())

    sa1 = sa(G, 10, 1000)
    g_out, y_list, ui_list = sa1.simulated_annealing("number of edges")
    
    #G = output1[0]
    print(g_out.number_of_edges())

    deg = [v for u,v in g_out.degree()]
    print(deg)

    plt.figure()
    nx.draw_networkx(g_out)#, pos=pos)
    # plt.show(block=False)

    print()
    print()
    # print(g_out.edges())

    # l1.append(G.number_of_edges())
    
    print(are_lc_equiv(G, g_out)[0])
    _, V = are_lc_equiv(G, g_out)
    # print(V)

    # print(g_out.number_of_edges())
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
#x = np.asarray(x)
#print(g_out.number_of_edges())
#plt.figure()
#plt.grid()
#plt.plot(x, l1, '-')
#plt.plot(x,x)
#plt.plot(x, l2, 'o')
#plt.plot(x, l3, '-')

#print(g_out.edges())
