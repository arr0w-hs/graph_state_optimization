#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:15:48 2024

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

from gen_bounded_deg import check_is_iso


graph_dir = os.path.join(dir_name, "er_results/mer_graphs_data/raw_data")
#plot_dir = os.path.join(dir_name, "bd_results_data/plots")

graph_list = []
for root, _, files in os.walk(graph_dir):
    for file in files:
        filename = str(file)

        prob_num = (filename.split("__")[1])
        prob_num = prob_num.split(".")[0]
        print(prob_num)


        if file == '.DS_Store':
            continue
        with open(os.path.join(graph_dir, file), 'rb') as f:
            data_dict_loaded = pickle.load(f)
            f.close()

        graph_list.append(list(data_dict_loaded.values()))

g_list = []
list_16 = []
for ele in graph_list:
    for elem in ele:

        if type(elem) == list:
            list_16 = list_16 + elem
        else:
            a = list(elem.values())
            for el in a:
                g_list.append((a[0]))

g_list.append(list_16)

graph_list = []
for ele in g_list:

    graph_list += ele

vert_list = np.arange(0.05, 1, step = 0.05)

non_iso_dict = {}
edge_dict = {}
g_list = []
edge_list = []
for elem in graph_list:
    if not check_is_iso(g_list, elem):
        g_list.append(elem)
        edge_list.append(elem.edges())

non_iso_dict[str(13)] = g_list
edge_dict[str(13)] = edge_list


with open(graph_dir+ "2all_mer" +'.pkl', 'wb') as f:  # open a text file
    pickle.dump(non_iso_dict, f)

edge_df = pd.DataFrame(edge_dict)
edge_df.to_csv(graph_dir+'out.csv', columns=edge_dict.keys())
# with open(graph_dir+"2csv.csv", "w", newline="") as f:
#     w = csv.DictWriter(f, edge_dict.keys())
#     w.writeheader()
#     w.writerow(edge_dict)
