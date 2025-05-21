#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:00:50 2024

@author: hsharma4
"""
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pathlib import Path
import argparse
#import psutil
import csv

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from optimizer.gsc.is_lc_equiv import are_lc_equiv
from gen_bounded_deg import check_is_iso
from optimizer.edm_sa import EDM_SimAnnealing as sa

graph_dir = os.path.join(dir_name, "bd_results/mer_graphs/raw_data")
input_graph = os.path.join(dir_name, "bd_results/bd_input_graph/")

graph_list = []
for root, _, files in os.walk(graph_dir):
    for file in files:
        if file == '.DS_Store':
            continue
        with open(os.path.join(graph_dir, file), 'rb') as f:
            data_dict_loaded = pickle.load(f)
            f.close()

        #out_df = pd.DataFrame(data_dict_loaded["out_dict"])


        graph_list.append(list(data_dict_loaded.values()))
        #print(data_dict_loaded["out_dict"].keys())

with open(os.path.join(input_graph, "100_2024-07-14_124819.pkl"), 'rb') as f:
    input_graph = pickle.load(f)
    f.close()

d_max = input_graph["d_max"]
g_data = input_graph["g_data"]



g_list = []
list_16 = []
for ele in graph_list:
    for elem in ele:
        #print(type(elem))
        if type(elem) == list:
            #print(len(elem))
            list_16 = list_16 + elem
        else:
            a = list(elem.values())
            for el in a:
                g_list.append((a[0]))



g_list.append(list_16)

graph_list = []
for ele in g_list:
    #print(len(ele))
    graph_list += ele

#print(len(graph_list))
vert_list = np.arange(6, 17, step = 1)
#print(vert_list)



graph_dict = {}

for ver in vert_list:
    g_temp = []

    for graph in graph_list:
        if graph.number_of_nodes() == ver:
            g_temp.append(graph)
    graph_dict[str(ver)] = g_temp

list_a = []
for i in range(10,11):
    i = i+6
    g_inp_list = g_data[str(i)]
    g_out = graph_dict[str(i)]

    print(i)
    for j, in_graph in enumerate(g_inp_list):
        #print(j)

        #sa1 = sa(in_graph, 100, 100)
        #g_out, x_list, ui_list = sa1.simulated_annealing("number of edges")
        print(len(g_out))
        for k, graphs in enumerate(g_out):

            #print(j)
            flag, _ = are_lc_equiv(in_graph, graphs)
            if flag:
                # print(j)
                list_a.append(j)
                g_out.remove(graphs)
                continue

print(np.sort(list_a))
list_100 = np.arange(0,100,1)
print(np.array_equal(list_100, list_a))
        # flag, _ = are_lc_equiv(in_graph, g_out[j])
        # print(flag)
        # if not flag:
        #     print(j, "sg")

# non_iso_dict = {}
# edge_dict = {}
# for key, value in graph_dict.items():
#     g_list = []
#     edge_list = []
#     for elem in list(value):
#         if not check_is_iso(g_list, elem):
#             g_list.append(elem)
#             edge_list.append(elem.edges())
#     #print(len(g_list))
#     non_iso_dict[key] = g_list
#     edge_dict[key] = [edge_list]

# with open(graph_dir+ "2all_mer" +'.pkl', 'wb') as f:  # open a text file
#     pickle.dump(non_iso_dict, f)

# edge_df = pd.DataFrame(edge_dict)
# edge_df.to_csv(graph_dir+'out.csv', columns=edge_dict.keys())

# # with open(graph_dir+"2csv.csv", "w", newline="") as f:
# #     w = csv.DictWriter(f, edge_dict.keys())
# #     w.writeheader()
# #     w.writerow(edge_dict)
