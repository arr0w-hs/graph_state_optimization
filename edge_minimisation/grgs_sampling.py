#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:39:10 2025

@author: hsharma4
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import scipy as sc
from networkx.algorithms import bipartite
sys.path.append(os.path.dirname(__file__))

dir_name = os.path.dirname(__file__)
sys.path.append('..')
from gso.edm_sa import EDM_SimAnnealing as sa
#from gso.gsc.is_lc_equiv import are_lc_equiv
from itertools import zip_longest, combinations
from networkx.algorithms import isomorphism as iso
from random import sample

def add_leaves(ingraph):
    outgraph = ingraph.copy()
    n = outgraph.number_of_nodes()

    for i in range(n):
        outgraph.add_node(n+i)
        outgraph.add_edge(i, n+i)

    return outgraph

def rank_over_F2(matrix):
    """
    Calculate the rank of a matrix over F2 (binary field).
    
    Parameters:
    matrix (list of lists or np.ndarray): Input matrix over F2.
    
    Returns:
    int: The rank of the matrix over F2.
    """
    # Ensure the matrix is a NumPy array with dtype int
    matrix = np.array(matrix, dtype=int) % 2
    rows, cols = matrix.shape
    rank = 0
    row_index = 0

    for col_index in range(cols):
        # Find a row with a leading 1 in the current column
        for i in range(row_index, rows):
            if matrix[i, col_index] == 1:
                # Swap the current row with the row_index row
                matrix[[row_index, i]] = matrix[[i, row_index]]
                break
        else:
            # No pivot found, continue to the next column
            continue
        
        # Eliminate all other 1s in the current column
        for i in range(rows):
            if i != row_index and matrix[i, col_index] == 1:
                matrix[i] ^= matrix[row_index]

        # Move to the next row and increase rank
        row_index += 1
        rank += 1

        # Stop if we've processed all rows
        if row_index >= rows:
            break

    return rank


"""
[(0, 9), (0, 10), (0, 11), (1, 8), (1, 9), (1, 11), (1, 12), 
 (1, 13), (2, 7), (2, 8), (2, 10), (3, 7), (3, 9), (3, 10), 
 (3, 11), (3, 12), (3, 13), (4, 7), (4, 8), (4, 10), (5, 8), 
 (5, 10), (5, 12), (5, 13), (6, 8), (6, 10), (6, 12), (6, 13)]
"""

"""
[[1 0 1 0 1 0 1]
 [0 1 1 0 1 1 0]
 [1 1 0 0 0 1 1]
 [0 0 0 1 1 1 1]
 [1 1 0 1 1 0 0]
 [0 1 1 1 0 0 1]
 [1 0 1 1 0 1 0]]
"""

k = 3
n = int(np.ceil(k/0.5)+1)
#n = 20
print(n)
avg_rank_list = []
graph_list = []
gen_list = []


prob = 0.9

for i in range(10000):
    # g_mat = [[1,0,1,1,1,0,0], [1,1,1,0,0,1,0], [0,1,1,1,0,0,1]]
    # g_mat = [[1,0,0,1,1,0], [0,1,0,1,0,1], [0,0,1,0,1,1]]
    # g_mat = np.asarray(g_mat)
    # n=6
    # k=3
    g_mat = np.random.randint(0, 2, (k,n))


    bi_adj = np.matmul(g_mat.transpose(), g_mat) % 2
    if rank_over_F2(bi_adj) !=k:
        continue
    flag = False
    for ele in bi_adj:
        if any(ele) != True:
            flag = True

    if flag:
        continue

    gen_list.append(g_mat)
    graph_list.append(bi_adj)
    rank_list = []

    # ini_rank = rank_over_F2(bi_adj)

    for it in range(1000):
        """it is the number of iterations"""

        row_list = [i for i, ele in enumerate(np.random.rand(n)) if ele < prob]
        col_list = [i for i, ele in enumerate(np.random.rand(n)) if ele < prob]

        # print(row_list, col_list)
        # print(bi_adj)
        bi_adj1 = []
        for i in row_list:
            for j in col_list:
                bi_adj1.append(bi_adj[i,j])
        bi_adj1 = np.reshape(bi_adj1, (len(row_list),len(col_list)))
        # print(bi_adj1)

        # rank_list.append(np.linalg.matrix_rank(bi_adj1))
        rank_list.append(rank_over_F2(bi_adj1))

    # avg_rank_list.append(np.average(rank_list))
    # rank_list = [1 for ele in rank_list if ele == ini_rank]
    # avg_rank_list.append( np.sum(rank_list))
    avg_rank_list.append(np.average(rank_list))

maxarg = np.argmax(avg_rank_list)
# # print(maxarg)
# print(graph_list[maxarg], "biadj mat")
print(avg_rank_list[maxarg], "avg rank over iterations")
# """choosing the graph with maximum value of average rank"""

bi_adj = graph_list[maxarg]

bi_adj = sc.sparse.csr_matrix(bi_adj)


G = nx.Graph()
G = bipartite.from_biadjacency_matrix(bi_adj)
sa1 = sa(G, 10000, 100)
G2, _, lc_loc1 = sa1.simulated_annealing("number of edges")

plt.figure()
nx.draw_networkx(G)#, pos=pos)
plt.draw()
plt.show(block = False)

G = add_leaves(G)
sa1 = sa(G, 10000, 100)
G2, _, lc_loc1 = sa1.simulated_annealing("number of edges")

plt.figure()
nx.draw_networkx(G2)#, pos=pos)
plt.draw()
plt.show(block = False)
