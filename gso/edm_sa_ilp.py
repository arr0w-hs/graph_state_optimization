#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:30:13 2024

@author: hsharma4
"""

import time
from gso.edm_sa import edm_sa
from gso.edm_ilp import edm_ilp


def edm_sa_ilp(G_in, k_max, temp):
    """
    function to find the MER for a given graph
    using SA+ILP and ILP. this returns the statistics of
    different algorithms
    inputs: 
        (1) graph to be minimised
        (2) k_max: maximum number of iterations
        (3) temp: initial temperature of the graph

    outputs:
        (1) output graph from SA+ILP
        (2) output graph from SA
        (3) number of edges in graph obtained from SA
        (4) number of edges in graph obtained from SA+ILP
        (5) runtime of SA+ILP
        (6) runtime of ILP

    """
    time1 = time.time()
    G_sa, y_list, _ = edm_sa(G_in, k_max, temp)
    sa_edges = (G_sa.number_of_edges())

    # print(len(G.edges()))
    G_final, sa_ilp_edges, ilp_runtime = edm_ilp(G_sa, draw=False)

    time2 = time.time()

    sa_ilp_runtime = time2-time1
    return G_final, G_sa, sa_edges, sa_ilp_edges, sa_ilp_runtime, ilp_runtime
