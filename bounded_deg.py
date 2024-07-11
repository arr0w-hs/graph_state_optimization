#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:40:46 2024

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
import time
#import mosek

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from edm_sa_ilp import edm_sa_ilp






if __name__ == "__main__":
    
    p = 0.6
    cl_list = []
    edge_data = []
    vert_list = []
    
    sa_ilp_data = []
    sa_data = []
    g_data = []
    gout_data = []
    runtime_data = []

    y_list = []
    ui_list = []

    edge_avg = []
    sa_ilp_avg = []
    sa_avg = []

    sa_min = []

    n = 10
    p_list = []
    max_k = 10*n
    temp_initial = 100
    """sample size is the number of graph sampled for each (n,p)"""
    sample_size = 1
    
    
    for j in range(2):
        
        #print(j)
        #p = 0.05*(j+1)
        p = 0.1*(j+1)
        p_list.append(p)

        edge_list = []
        sa_ilp_list = []
        sa_list = []
        gout_list = []
        g_list = []
        runtime_list = []
        
        flag = 1
        for i in range(sample_size):
            #print(i)
            
            #if !nx.is_connected(G):
            #    continue
            #while flag = 1:
            G = nx.erdos_renyi_graph(n, p)
            output = edm_sa_ilp(G, max_k, temp_initial)
            
            print(j, i, output[2], output[3])
            #print()
            #print(output[4])
            
            edge_list.append(G.number_of_edges())
            
            g_list.append(G)
            gout_list.append(output[0])
            sa_list.append(output[2])
            sa_ilp_list.append(output[3])
            runtime_list.append(output[4])


        edge_data.append(edge_list)
        sa_ilp_data.append(sa_ilp_list)
        sa_data.append(sa_list)
        g_data.append(g_list)
        gout_data.append(gout_list)
        runtime_data.append(runtime_list)
        
        edge_avg.append(np.mean(edge_list))
        sa_ilp_avg.append(np.mean(sa_ilp_list))
        sa_avg.append(np.mean(sa_list))
        

    data_dict = {
        "n": n,
        "prob_list": p_list,
        "edge_data": edge_data,
        "sa_ilp_data": sa_ilp_data,
        "sa_data": sa_data,
        "runtime_data": runtime_data,
        }

    graph_dict = {
        "n": n,
        "g_data": g_data,
        "sample size": sample_size,
        "gout_data": gout_data,
        }

    metadata_dict = {
        "num of vertex": n,
        "sample size": sample_size,
        "prob_list": p_list,
        }


    ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
    date_str = str(ts.date())

    time_str = ts.time()
    time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
    print(time_str)
    data_directory = os.path.join(dir_name+"/data", date_str+"_sa_ilp/")
    graph_directory= os.path.join(dir_name+"/graphs", "n="+str(n)+"/")
    plots_directory = os.path.join(dir_name+"/plots", date_str+"_sa_ilp/")

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

    with open(graph_directory+ str(sample_size)
              + "_"+ date_str + "_" + time_str +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(graph_dict, f)

    with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
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
    