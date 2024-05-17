#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:42:28 2024

@author: hsharma4
"""


import sys
import os

import pickle

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams.update({'font.size': 12})

from local_complementation import *


    
    
def vertex_choice(inp_graph, transition_cutoff, k_max):
    """metric with multiplication of clustering with degree"""
    degree_list = [val for (node, val) in inp_graph.degree()]
    clustering_dict = nx.clustering(inp_graph)
    clustering_val = list(clustering_dict.values())
    new_metric_val = np.multiply(degree_list, clustering_val)
    
    if np.sum(new_metric_val) != 0:
        new_metric_val = new_metric_val/np.sum(new_metric_val)
    
    """find all the unique elemetns in the array
    and randomly choose an element form the vertices which are 
    larger than the value of cutoff_arg at that point
    """
    
    unique_elements = np.unique(new_metric_val)
    #cutoff_arg = np.ceil((1-1/transition_cutoff)*len(unique_elements))
    if k_max != 0:
        cutoff_arg = np.ceil((transition_cutoff/k_max)*len(unique_elements))
    else:
        raise Exception("k_max is 0")
    
    if cutoff_arg < len(unique_elements):
        chosen_arg = np.random.randint(cutoff_arg, len(unique_elements))
    else:
        chosen_arg = len(unique_elements) - 1
    
    vertices = [index for index in np.flatnonzero(new_metric_val) 
                if new_metric_val[index] == unique_elements[chosen_arg]]

    if len(vertices) == 0:
        vertices = np.flatnonzero(clustering_val == np.max(clustering_val))
    
    vertex = np.random.choice(vertices)
    
    return vertex


def energy_func(inp_graph):
    

    energy = inp_graph.number_of_edges()
    
    return energy


def sa(inp_graph, k_max, initial_temp):
    y = energy_func(inp_graph)
    y_best = y
    x = 0
    temp = initial_temp
    transition_cutoff = 1
    g_best = inp_graph
    g = inp_graph
    
    edge_list = []
    best_list = []
    for k in range(k_max):
        #print(y_best)
        #print(transition_cutoff, "tr cutoff")
        x_new = vertex_choice(g, transition_cutoff, k_max)
        #x_new = np.random.randint(0, n)
        g_new = local_complementation(g, x_new)
        y_new = energy_func(g_new)
        
        dy = y_new - y
        #print(dy)
        if dy <= 0 or np.random.uniform(0,1,1) < np.exp(-1*dy/temp):
            y = y_new
            g = g_new
            
            
        
        if y_new < y_best:
            g_best = g_new
            y_best = y_new
        #print(np.log(k+1))
        temp = initial_temp*np.log(2)/(np.log(k+2))
        #temp = initial_temp/(k+2)
        transition_cutoff = (k+1)
        #transition_cutoff = (k_max)
        #print(transition_cutoff)
        print(temp)
        edge_list.append(y)
        best_list.append(y_best)
        
    #print(len(edge_list))
    return g_best, edge_list, best_list



p = 0.3
cl_list = []
edge_data = []
vert_list = []
nm_data = []
sa_data = []
g_data = [] 

y_list = []
ui_list = []

edge_avg = []
nm_avg = []
sa_avg = []

n = 15
x = []
k_max = 10*n
initial_temp = 100
"""sample size is the number of graph sampled for each (n,p)"""
sample_size = 1000

for j in range(1):
    
    print(j)
    p = 0.05*(j+1)
    p = 0.75
    x.append(p)
    
    edge_list = []
    nm_list = []
    sa_list = []
    g_list = []
    G = nx.fast_gnp_random_graph(n, p)
    for i in range(1):
        #print(i)
        
        min_nm_edge, min_nm_graph = minimisation_new_metric(G)
        g_out, y_list, ui_list = sa(G, k_max, initial_temp)
        print(np.min(ui_list))
        g_list.append(G)
        edge_list.append(energy_func(G))
        sa_list.append(energy_func(g_out))
        nm_list.append(min_nm_edge)
        
        xx = np.linspace(0, len(y_list), len(y_list))
        plt.figure()
        plt.grid()
        plt.plot(xx, y_list, xx, ui_list)
        plt.show(block = False)
    
    edge_data.append(edge_list)
    nm_data.append(nm_list)
    sa_data.append(sa_list)
    g_data.append(g_list)
    
    edge_avg.append(np.mean(edge_list))
    nm_avg.append(np.mean(nm_list))
    sa_avg.append(np.mean(sa_list))
    #g_data.append(g_list)
print(nm_avg, sa_avg)
plt.figure()
plt.grid()
#plt.plot(x, num_edges_list, '-o', x, nm_list, '-o', x, sa_list, '-o')
plt.plot(x, edge_avg, '-o',  x, nm_avg, '-o',  x, sa_avg, '-o')
plt.title("n="+str(n))
#plt.plot(x, cl_avg, x, nm_avg, x, tr_avg, x, edge_avg)
plt.ylabel('Number of edges')
plt.xlabel('Graph')
plt.legend(["Initial edges", "New metric", "Simulated annealing"])
#plt.savefig(plots_directory + time_str + "_n="+str(n)+"_performance" + ".png", dpi=1000, format="png", bbox_inches = 'tight')









print(lauda)

data_dict = {
    "n": n,
    "prob_list": x,
    "edge_data": edge_data,
    "nm_data": nm_data,
    "sa_data": sa_data,  
    }

graph_dict = {
    "n": n,
    "g_data": g_data,
    "sample size": sample_size,
    }
    
metadata_dict = {
    "num of vertex": n,
    "sample size": sample_size,
    }

x = np.linspace(0, len(y_list), len(y_list))
plt.figure()
plt.grid()
plt.plot(x, y_list)
print(lauda)

ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
date_str = str(ts.date())

time_str = ts.time()
time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
print(time_str)
data_directory = os.path.join(dir_name+"/data", date_str+"_simulated_annealing/")
graph_directory= os.path.join(dir_name+"/graphs", "n="+str(n)+"/")
plots_directory = os.path.join(dir_name+"/plots", date_str+"_simulated_annealing/")

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

with open(graph_directory+ str(sample_size) + "_"+ date_str + "_" + time_str +'.pkl', 'wb') as f:  # open a text file
    pickle.dump(graph_dict, f)

with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
    f.write(str(metadata_dict))
    f.close()

#x = np.linspace(0, itera, itera)
plt.figure()
plt.grid()
#plt.plot(x, num_edges_list, '-o', x, nm_list, '-o', x, sa_list, '-o')
plt.plot(x, edge_avg,  x, nm_avg,  x, sa_avg)
plt.title("n="+str(n))
#plt.plot(x, cl_avg, x, nm_avg, x, tr_avg, x, edge_avg)
plt.ylabel('Number of edges')
plt.xlabel('Graph')
plt.legend(["Initial edges", "New metric", "Simulated annealing"])
#plt.savefig(plots_directory + time_str + "_n="+str(n)+"_performance" + ".png", dpi=1000, format="png", bbox_inches = 'tight')

