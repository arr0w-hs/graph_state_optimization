#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:10:09 2024

@author: hsharma4
"""

import sys
import os
# from pathlib import Path


#os.chdir(dir_name)

import numpy as np
# import time as time
# import matplotlib.pyplot as plt
# from math import cos, sin, pi
import networkx as nx

import scipy as sc
from networkx.algorithms import bipartite
sys.path.append(os.path.dirname(__file__))
# dir_name = os.path.dirname(__file__)

dir_name = os.path.dirname(__file__)
sys.path.append('..')
# import random as random
# from optimizer.edm_sa_ilp import edm_sa_ilp
from optimizer.edm_sa import EDM_SimAnnealing as sa
# from optimizer.ILP_minimize_edges import minimize_edges as ilp_minimize_edges
from optimizer.gsc.is_lc_equiv import are_lc_equiv
from itertools import zip_longest, combinations
from networkx.algorithms import isomorphism as iso


def graph_equiv(G1, G2):

    g_list = [G1, G2]
    # g_list = [nx.convert_node_labels_to_integers(ele)#, ordering= "sorted")
              # for ele in g_list]
    g_list = [nx.to_numpy_array(ele) for ele in g_list]

    return np.array_equiv(g_list[0], g_list[1])


def local_comp(in_graph, vert):
    out_graph = in_graph
    neigh = [ne for ne in out_graph.neighbors(vert)]
    neigh_combinations = combinations(neigh, 2)
    # print(neigh_combinations, vert, "nv")

    for u,v in neigh_combinations:
        # print(u,v)
        # print(G.has_edge(u,v))
        if out_graph.has_edge(u,v):
            out_graph.remove_edge(u,v)
        else:
            out_graph.add_edge(u,v)

    return out_graph

def rm_double_lc(local_comp_list):
    flag = 0
    diff = 1
    limit = 5*len(local_comp_list)
    count = 0
    while diff > 0 and count < limit:
        res = []
        count+=1
        for i, j in (zip_longest(local_comp_list, local_comp_list[1:])):
            if flag != 1:
                if i == j:
                    flag = 1
                    continue
                else:
                    res.append(i)
            else:
                flag = 0
        diff = len(local_comp_list) - len(res)
        local_comp_list = res

    return local_comp_list

def find_iso_graphs(diff_list, val_list):
    # print(diff_list)
    # print(val_list)

    final_list = []


    count = 0

    zipped = (zip(diff_list, val_list))
    zipped  = list(reversed(sorted(zipped, key = lambda x: x[0])))


    # zip_max = [ele for ele in zipped if ele[0] == zipped[0][0]]
    # print(zip_max, "zipmax")


    while len(zipped) > 0 and count < 10*len(diff_list):
        count += 1
        other_list = []
        diff, val = zipped[0]
        vval, uval = val
        final_list.append(zipped[0])


        for differ, value in zipped:
    
            if value == val:
                continue
    
            vval1, uval1 = value
    
            if vval1 < uval and uval1 < uval:
                other_list.append((differ, value))
            elif uval1 > vval and vval1 > vval:
                other_list.append((differ, value))
        zipped = other_list

    # print(other_list)
    print()
    print(final_list)
    print()


    return final_list

def remove_isomorphism_path(mer_graph, local_comp_list):

    graph_list = []
    G3 = mer_graph
    sa2 = sa(G3, 100, 100)
    for ele in (local_comp_list):
        G3 = sa2.local_complementation(G3, ele)
        graph_list.append(G3)

    graph_list = list(zip(range(len(graph_list)), graph_list))
    graph_comb = combinations(graph_list, 2)
    diff_list = []
    val_list = []
    for u, v in graph_comb:

        if nx.is_isomorphic(u[1], v[1]):
            diff_list.append(v[0] - u[0])
            val_list.append((v[0], u[0]))


    if len(diff_list) >0:
        iso_locations = find_iso_graphs(diff_list, val_list)

        vals = val_list[np.argmax(diff_list)]
        local_comp_list = [ele for i, ele in enumerate(local_comp_list)
                           if i <= vals[1] or i > vals[0]]
        print(vals, "vals")
        gr1 = graph_list[vals[1]][1]
        gr2 = graph_list[vals[0]][1]
        GM = iso.GraphMatcher(gr2,gr1)
        iso_mapping = GM.mapping


        GM = iso.GraphMatcher(gr1, gr2)
        GM.is_isomorphic()
        iso_mapping = GM.mapping


    else:
        vals = []
        iso_mapping = {}

    # print(iso_mapping, 'isomaping')
    return local_comp_list, vals, iso_mapping


def reconstr_ini_graph(mer_graph, local_comp_list, uval, iso_mapping):
    out_graph = mer_graph
    _, uval = uval
    sa2 = sa(out_graph, 100, 100)

    for i, ele in enumerate(local_comp_list):

        if i==uval:

            sa2 = sa(out_graph, 100, 100)
            out_graph = sa2.local_complementation(out_graph, ele)
            out_graph = nx.relabel_nodes(out_graph, iso_mapping)

        else:

            sa2 = sa(out_graph, 100, 100)
            out_graph = sa2.local_complementation(out_graph, ele)

    # print(out_graph.number_of_edges(), "edges after constructing back")


    return out_graph, iso_mapping




if __name__ == "__main__":
    k = 2
    n = int(np.ceil(k/0.5)+1)
    #n = 20
    # print(n)

    for i in range(1):
        g_mat = np.random.randint(0, 2, (k,n))

        bi_adj = np.matmul(g_mat.transpose(), g_mat) % 2
        # print(bi_adj)
        bi_adj = sc.sparse.csr_matrix(bi_adj)
        # print(bi_adj)
        G = nx.Graph()
        G = bipartite.from_biadjacency_matrix(bi_adj)

        # G = nx.erdos_renyi_graph(2*n, 0.75)


        if not nx.is_connected(G):
            continue
        # pos = nx.bipartite_layout(G, list(range(n)))
        # plt.figure()
        # nx.draw_networkx(G)#, pos=pos)
        # plt.draw()
        # plt.show(block = False)

        # print(G.number_of_edges(), "initial edges")
        sa1 = sa(G, 100, 100)
        G2, _, lc_loc = sa1.simulated_annealing("number of edges")
        # print(G2.number_of_edges(), "min edges")

        lc_loc = rm_double_lc(lc_loc)
        lc_loc, uv_vals, mapping = remove_isomorphism_path(G2, lc_loc)



        if len(uv_vals) >0:
            outg, isomap = reconstr_ini_graph(G2, lc_loc, uv_vals, mapping)
            flag = nx.is_isomorphic(outg, G)#, "after constr is isomorphic")
            print(nx.is_isomorphic(outg, G), "after constr is isomorphic")
            print(len(lc_loc))
            # print(uv_vals)
            if not flag:
                print(isomap)

        else:
            # print("none")
            continue
