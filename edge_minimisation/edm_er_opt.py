#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:49 2024

@author: hsharma4
code for analysing edge_minimisation on ER graphs
"""

import sys
import os
from pathlib import Path
#dir_name = os.path.dirname(__file__)
#os.chdir(dir_name)
sys.path.append('..')

import pickle
import pandas as pd

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from gso.edm_sa_ilp import edm_sa_ilp

"""importing the generated bounded deg graphs"""
graph_filename = '/13_100_20_2024-07-16_162924.pkl'
with open(dir_name+'/er_results/er_graphs'+graph_filename, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

p_list = data_dict_loaded["p_list"]
g_data = data_dict_loaded["g_data"]

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

print(g_data.keys())


for i, ele in enumerate(p_list):
    n = 13

    g_inp_list = g_data[str(ele)]

    for G in g_inp_list:

        output = edm_sa_ilp(G, max_k, temp_initial)
        edge_list.append(G.number_of_edges())

        gout_list.append(output[0])
        sa_list.append(output[2])
        sa_ilp_list.append(output[3])
        runtime_sailp_list.append(output[4])


        output_ilp = 0#ilp_minimize_edges(G)#, max_k, temp_initial)
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
    "input_file": graph_filename,
    }

ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
date_str = str(ts.date())

time_str = ts.time()
time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
print(time_str)
data_directory = os.path.join(dir_name+"/er_results/results", date_str+"_sa_ilp_er/")
graph_directory= os.path.join(dir_name+"/er_results/mer_graphs", "n="+str(n)+"/")
plots_directory = os.path.join(dir_name+"/er_results/plots", date_str+"_sa_ilp_er")

date_folder = Path(data_directory)
if not date_folder.exists():
    os.mkdir(data_directory)

graph_folder = Path(graph_directory)
if not graph_folder.exists():
    os.mkdir(graph_directory)

plot_folder = Path(plots_directory)
if not plot_folder.exists():
    os.mkdir(plots_directory)

with open(data_directory+ time_str +'.pkl', 'wb') as f:  # open a text file
    pickle.dump(data_dict, f)
with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
    f.write(str(metadata_dict))
    f.close()

with open(graph_directory
          + "_"+ date_str + "_" + time_str +'.pkl', 'wb') as f:  # open a text file
    pickle.dump(graph_dict, f)

with open(graph_directory + time_str +'_metadata.txt', mode="w") as f:
    f.write(str(metadata_dict))
    f.close()
