#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:49 2024

@author: hsharma4
code for analysing edge_minimisation on ER graphs
"""

from pathlib import Path
import networkx as nx
from graphstate_opt.edm_sa_ilp import edm_sa_ilp

import pandas as pd





out_dict = {}
gout_dict = {}

max_k = 100
temp_initial = 100

prob_list = []
edge_list = []
sa_ilp_list = []
sa_list = []
gout_list = []
g_list = []
runtime_sailp_list = []
runtime_ilp_list = []

for _ in range(5):
    G = nx.erdos_renyi_graph(5, 0.8)

    output = edm_sa_ilp(G, max_k, temp_initial)
    edge_list.append(G.number_of_edges())

    gout_list.append(output[0])
    sa_list.append(output[2])
    sa_ilp_list.append(output[3])
    runtime_sailp_list.append(output[4])


    output_ilp = 0#ilp_minimize_edges(G)#, max_k, temp_initial)
    runtime_ilp_list.append(output_ilp)
    prob_list.append(0.8)

out_dict["prob_val"] = prob_list
out_dict["edge"] = edge_list
out_dict["sa_edge"] = sa_list
out_dict["sa_ilp_edge"] = sa_ilp_list
out_dict["rt_sailp"] = runtime_sailp_list
out_dict["rt_ilp"] = runtime_ilp_list


print(out_dict)