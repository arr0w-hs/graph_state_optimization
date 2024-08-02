#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:14 2024

@author: hsharma4
code for edge minimisation of bounded degree graphs
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pathlib import Path

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from edm_sa_ilp import edm_sa_ilp
from edm_ilp import ilp_minimize_edges

"""importing the generated bounded deg graphs"""

graph_location = '/100_2024-07-14_124819.pkl'
with open(dir_name+'/bounded_deg_graphs'+graph_location, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

print(data_dict_loaded.keys())


d_max = data_dict_loaded["d_max"]
g_data = data_dict_loaded["g_data"]

#print(type(g_data))
print(g_data.keys())
for i in range(20):
    i = i+6
    g_inp_list = g_data[str(i)]
    print(len(g_inp_list))




#g_dt = pd.DataFrame(data = g_data)
#print(g_dt.dtypes)
#print(n_list)
#print(adf)


out_dict = {}
gout_dict = {}


edge_avg = []
sa_ilp_avg = []
sa_avg = []

sa_min = []

n = 10
p_list = []
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
#print(stopp)
for i in range(1):

    n= 6+i
    print(n)






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

    #edge_avg.append(np.mean(edge_list))
    #sa_ilp_avg.append(np.mean(sa_ilp_list))
    #sa_avg.append(np.mean(sa_list))


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
plots_directory = os.path.join(dir_name+"/plots", date_str+"_sa_ilp_bd/")

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

"""
x = np.linspace(0, sample_size, sample_size)
plt.figure()
plt.grid()
#plt.plot(x, num_edges_list, '-o', x, nm_list, '-o', x, sa_list, '-o')
#plt.plot(x, edge_avg, x, sa_ilp_avg,  x, sa_min)
plt.plot(x, edge_list, x, sa_ilp_list,  x, sa_list)
plt.title("n="+str(n))
#plt.plot(x, cl_avg, x, nm_avg, x, tr_avg, x, edge_avg)
plt.ylabel('Number of edges')
plt.xlabel('Graph')
plt.legend(["Initial edges", "New metric", "Simulated annealing"])
#plt.savefig(plots_directory+time_str+"_n="+str(n)+"_performance" + ".png",
#            dpi=1000, format="png", bbox_inches = 'tight')

"""
