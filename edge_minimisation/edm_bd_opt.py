#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:14 2024

@author: hsharma4
code for edge minimisation of bounded degree graphs
"""

import sys
import os
from pathlib import Path
dir_name = os.path.dirname(__file__)
os.chdir(dir_name)
sys.path.append('..')

import pickle
import pandas as pd

from optimizer.edm_sa_ilp import edm_sa_ilp
from optimizer.edm_ilp import ilp_minimize_edges

"""importing the generated bounded deg graphs"""
graph_location = '/100_2024-07-14_124819.pkl'
with open(dir_name+'/bounded_deg_graphs'+graph_location, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

d_max = data_dict_loaded["d_max"]
g_data = data_dict_loaded["g_data"]

for i in range(20):
    i = i+6
    g_inp_list = g_data[str(i)]


out_dict = {}
gout_dict = {}

sa_min = []

n = 10
max_k = 10*n
temp_initial = 100

vertex_list = []
edge_list = []
sa_ilp_list = []
sa_list = []
gout_list = []
g_list = []
runtime_sailp_list = []
runtime_ilp_list = []

for i in range(1):

    n= 6+i

    g_inp_list = g_data[str(n)]

    for G in g_inp_list:

        output = edm_sa_ilp(G, max_k, temp_initial)

        edge_list.append(G.number_of_edges())

        gout_list.append(output[0])
        sa_list.append(output[2])
        sa_ilp_list.append(output[3])
        runtime_sailp_list.append(output[4])


        output_ilp = ilp_minimize_edges(G)#, max_k, temp_initial)
        runtime_ilp_list.append(output_ilp[2])


        vertex_list.append(n)


    gout_dict[str(n)] = gout_list

out_dict["vertex_num"] = vertex_list
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
    }

ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
date_str = str(ts.date())

time_str = ts.time()
time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
print(time_str)
data_directory = os.path.join(dir_name+"/data", date_str+"_sa_ilp_bd/")
graph_directory= os.path.join(dir_name+"/graphs", "n="+str(n)+"/")

date_folder = Path(data_directory)
if not date_folder.exists():
    os.mkdir(data_directory)

graph_folder = Path(graph_directory)
if not graph_folder.exists():
    os.mkdir(graph_directory)

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
