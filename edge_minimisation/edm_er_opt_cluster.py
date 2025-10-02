#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:49 2024

@author: hsharma4
code for analysing Edge minimisation on ER graphs on cluster with argparse
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


plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from gso.edm_sa_ilp import edm_sa_ilp
from gso.edm_ilp import ilp_minimize_edges

parser = argparse.ArgumentParser(description="Set probability value")
parser.add_argument("--i", type=int, help="index of probability list")
args, _ = parser.parse_known_args()

"""importing the generated bounded deg graphs"""

graph_location = '/13_100_20_2024-07-16_162924.pkl'
with open(dir_name+'/er_results/er_graphs'+graph_location, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

print(data_dict_loaded.keys())


p_list = data_dict_loaded["p_list"]
g_data = data_dict_loaded["g_data"]

n=10
i = args.i

ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
date_str = str(ts.date())

time_str = ts.time()
time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
print(time_str)
data_directory = os.path.join(dir_name+"/er_results/results", date_str+"_sa_ilp_er/")
graph_directory= os.path.join(dir_name+"/er_results/mer_graphs", "n="+str(n)+"/")
plots_directory = os.path.join(dir_name+"/er_results/plots", date_str+"_sa_ilp_er/")

date_folder = Path(data_directory)
if not date_folder.exists():
    os.mkdir(data_directory)

graph_folder = Path(graph_directory)
if not graph_folder.exists():
    os.mkdir(graph_directory)

plot_folder = Path(plots_directory)
if not plot_folder.exists():
    os.mkdir(plots_directory)



out_dict = {}
gout_dict = {}

edge_avg = []
sa_ilp_avg = []
sa_avg = []

sa_min = []


max_k = 10*n
temp_initial = 100

prob_list = []
edge_list = []
sa_ilp_list = []
sa_list = []
gout_list = []
g_list = []
runtime_sailp_list = []
runtime_ilp_list = []

print(g_data.keys())

i = 10
print(i)
ele = p_list[i]

n= 10

g_inp_list = g_data[str(ele)]

for G in g_inp_list:

    output = edm_sa_ilp(G, max_k, temp_initial)
    edge_list.append(G.number_of_edges())

    gout_list.append(output[0])
    sa_list.append(output[2])
    sa_ilp_list.append(output[3])
    runtime_sailp_list.append(output[4])


    output_ilp = 0
    runtime_ilp_list.append(output_ilp)

    prob_list.append(ele)

    gout_dict[str(n)] = gout_list
    out_dict["prob_val"] = prob_list
    out_dict["edge"] = edge_list
    out_dict["sa_edge"] = sa_list
    out_dict["sa_ilp_edge"] = sa_ilp_list
    out_dict["rt_sailp"] = runtime_sailp_list
    out_dict["rt_ilp"] = runtime_ilp_list

    data_dict = {
        "out_dict": out_dict,
        }

    graph_dict = {
        "gout_dict": gout_dict,
        }

    metadata_dict = {
        "input_file": graph_location,
        "len_gout":len(gout_list),
        }

    with open(data_directory+ time_str
              + "__"+str(i) +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(data_dict, f)
    with open(data_directory + time_str+ "__"+str(i) +'_metadata.txt', mode="w") as f:
        f.write(str(metadata_dict))
        f.close()

    with open(graph_directory
              + "_"+ date_str + "_" + time_str
              +"__"+str(i) +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(graph_dict, f)

    with open(graph_directory + time_str + "__"+str(i) +'_metadata.txt', mode="w") as f:
        f.write(str(metadata_dict))
        f.close()
