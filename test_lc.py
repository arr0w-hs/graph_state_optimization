#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:08:23 2024

@author: hsharma4
"""
import sys
import os

import pickle

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams.update({'font.size': 12})

from base_lc_old import *

test = True
err = []
cl_data = []
nm_data = []
tr_data = []
edge_data = []
g_data = []

cl_avg = []
nm_avg = []
tr_avg = []
edge_avg = []

cl_avg1 = []
nm_avg1 = []
tr_avg1 = []
edge_avg1 = []

cl_avg2 = []
nm_avg2 = []
tr_avg2 = []
edge_avg2 = []

x = []
n = 10
sample_size = 1000
#p = 0.6
for j in range(100):
    print(j)
    p = 0.01+j*0.01
    #p = 0.7
    #p = 0.1*(j+1)
    x.append(p)
    edge_list = []
    nm_list = []
    cl_list = []
    tr_list = []

    edge_list1 = []
    nm_list1 = []
    cl_list1 = []
    tr_list1 = []

    edge_list2 = []
    nm_list2 = []
    cl_list2 = []
    tr_list2 = []

    cl_graph_list = []
    tr_graph_list = []
    nm_graph_list = []
    #it = 500*(j+1)
    g_list = []

    for i in range(sample_size):
        #print(i)
        G = nx.fast_gnp_random_graph(n, p)
        edge_list.append(G.number_of_edges())
        edge_list1.append(nx.total_spanning_tree_weight(G))#,method='lanczos'))
        g_list.append(G)
        min_cl_edge, min_cl_graph = minimisation_clustering(G)
        min_nm_edge, min_nm_graph = minimisation_new_metric(G)
        min_tr_edge, min_tr_graph = minimisation_triangle(G)

        cl_list.append(min_cl_edge)
        nm_list.append(min_nm_edge)
        tr_list.append(min_tr_edge)
        #number_of_spanning_trees

        cl_list1.append(nx.algebraic_connectivity(min_cl_graph,method='lanczos'))
        nm_list1.append(nx.algebraic_connectivity(min_nm_graph,method='lanczos'))
        tr_list1.append(nx.algebraic_connectivity(min_tr_graph,method='lanczos'))

        #cl_list1.append(nx.total_spanning_tree_weight(min_cl_graph))
        #nm_list1.append(nx.total_spanning_tree_weight(min_nm_graph))
        #tr_list1.append(nx.total_spanning_tree_weight(min_tr_graph))

        if min_nm_edge > G.number_of_edges():
            err.append(G)
            #print(min_nm_edge, G.number_of_edges())
            #plt.figure()
            #nx.draw_networkx(G)
            #plt.draw()
    #edge_std_max
    edge_data.append((edge_list))
    cl_data.append((cl_list))
    nm_data.append((nm_list))
    tr_data.append((tr_list))

    edge_avg.append(np.mean(edge_list))
    cl_avg1.append(np.mean(cl_list1))
    nm_avg1.append(np.mean(nm_list1))
    tr_avg1.append(np.mean(tr_list1))

    cl_avg.append(np.mean(cl_list))
    nm_avg.append(np.mean(nm_list))
    tr_avg.append(np.mean(tr_list))

    g_data.append(g_list)
print(len(err), "wrong moves")

#x = np.linspace(0, 1, 20)
plt.figure()
plt.grid()
#plt.plot(x, cl_avg, '-o', x, nm_avg, '-o', x, tr_avg, '-o')#, x, edge_avg, '-o')
#plt.yscale('log')
plt.title("n="+str(n))
plt.plot(x, cl_avg, x, nm_avg, x, tr_avg)#, x, edge_avg)
plt.ylabel('Average number of edges')
plt.xlabel('Probability')
plt.legend(["Clustering", "New metric", "# triangles", "edge"])
#plt.savefig(plots_directory + time_str + "_n="+str(n)+"_performance" + ".png", dpi=1000, format="png", bbox_inches = 'tight')

plt.figure()
plt.grid()
#plt.plot(x, cl_avg1, '-o', x, nm_avg1, '-o', x, tr_avg1, '-o')#, x, edge_avg, '-o')
#plt.yscale('log')
plt.title("n="+str(n))
plt.plot(x, cl_avg1, x, nm_avg1, x, tr_avg1)#, x, edge_avg)
#plt.ylabel('Spanning tree')
plt.ylabel('algebraic connectivity')
plt.xlabel('Probability')
plt.legend(["Clustering", "New metric", "# triangles", "edge"])

if not test:
    print('meow')
    data_dict = {
        "prob_list": x,
        "edge_data": edge_data,
        "cl_data": cl_data,
        "nm_data": nm_data,
        "tr_data": tr_data,
        "g_data": g_data,
        }

    ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
    date_str = str(ts.date())

    time_str = ts.time()
    time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
    print(time_str)
    data_directory = os.path.join(dir_name+"/data", date_str+"_local_complementation/")
    plots_directory = os.path.join(dir_name+"/plots", date_str+"_local_complementation/")

    date_folder = Path(data_directory)
    if date_folder.exists():
        print("date folder exists")
    else:
        os.mkdir(data_directory)

    plot_folder = Path(plots_directory)
    if plot_folder.exists():
        print("plot folder exists")
    else:
        os.mkdir(plots_directory)


    with open(data_directory+ time_str +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(data_dict, f)


    metadata_dict = {
        "num of vertex": n,
        "sample size": sample_size,
        }

    with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
        f.write(str(metadata_dict))
        f.close()
