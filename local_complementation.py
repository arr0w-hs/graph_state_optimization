#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:10:24 2024

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

def local_complementation(inp_graph, vert):
    adj_mat = nx.to_numpy_array(inp_graph)
    row = adj_mat[vert]
    adj_mat = (adj_mat + np.outer(row, row))
    np.fill_diagonal(adj_mat, 0)
    adj_mat = adj_mat%2

    return nx.from_numpy_array(adj_mat)

def max_clustered_nodes(inp_graph):
    clustering_dict = nx.clustering(inp_graph)
    clustering_val = list(clustering_dict.values())
    #print(clustering_val, "clustering val")
    vertices = np.flatnonzero(clustering_val == np.max(clustering_val))

    return vertices

def max_triangle_nodes(inp_graph):
    triangles_dict = nx.triangles(inp_graph)
    triangles_val = list(triangles_dict.values())
    #print(triangles_val, "triangles val")
    vertices = np.flatnonzero(triangles_val == np.max(triangles_val))

    return vertices

def high_clustered_nodes(inp_graph):
    clustering_dict = nx.clustering(inp_graph)
    clustering_val = list(clustering_dict.values())
    vertices = [index for index in np.flatnonzero(clustering_val) if clustering_val[index] >= 0.5]

    if len(vertices) == 0:
        vertices = np.flatnonzero(clustering_val == np.max(clustering_val))

    return vertices

def new_metric_nodes(inp_graph):
    """metric with multiplication of clustering with degree"""

    degree_list = [val for (node, val) in inp_graph.degree()]

    clustering_dict = nx.clustering(inp_graph)
    clustering_val = list(clustering_dict.values())

    new_metric_val = np.multiply(degree_list, clustering_val)
    #print(new_metric_val, "new_metric_val")
    #if np.sum(new_metric_val) !=0:
    #    print(new_metric_val/np.sum(new_metric_val), "normalized")
    vertices = np.flatnonzero(new_metric_val == np.max(new_metric_val))
    if len(vertices) == 0:
        vertices = np.flatnonzero(clustering_val == np.max(clustering_val))

    return vertices


def apply_lc_clustering(inp_graph):
    update_graph_list = []
    avg_clustering_list = []
    flagg = 1
    vert_list = max_clustered_nodes(inp_graph)

    for i, vert in enumerate(vert_list):
        updated_graph = local_complementation(inp_graph, vert)
        avg_clustering = nx.average_clustering(updated_graph)

        update_graph_list.append(updated_graph)
        avg_clustering_list.append(avg_clustering)
    """vertex with most effect (which reduces avg clustering)
        is opt_vert"""
    opt_vert = np.argmin(avg_clustering_list)
    op_graph = update_graph_list[opt_vert]

    if nx.average_clustering(inp_graph) <= nx.average_clustering(op_graph):
        op_graph = inp_graph
        flagg = 0
    return op_graph, vert_list[opt_vert], flagg

def apply_lc_triangles(inp_graph):
    update_graph_list = []
    avg_clustering_list = []
    vert_list = max_triangle_nodes(inp_graph)
    flagg = 1

    for i, vert in enumerate(vert_list):
        updated_graph = local_complementation(inp_graph, vert)
        avg_clustering = nx.average_clustering(updated_graph)

        update_graph_list.append(updated_graph)
        avg_clustering_list.append(avg_clustering)
    """vertex with most effect (which reduces avg clustering)
        is opt_vert"""

    triangles_dict = nx.triangles(inp_graph)
    triangles_val = list(triangles_dict.values())


    opt_vert = np.argmin(avg_clustering_list)
    op_graph = update_graph_list[opt_vert]

    if nx.average_clustering(inp_graph) <= nx.average_clustering(op_graph):
        op_graph = inp_graph
        flagg = 0

    return op_graph, vert_list[opt_vert], flagg

def apply_high_lc_clustering(inp_graph):
    update_graph_list = []
    avg_clustering_list = []
    flagg = 1
    vert_list = high_clustered_nodes(inp_graph)

    for i, vert in enumerate(vert_list):
        updated_graph = local_complementation(inp_graph, vert)
        avg_clustering = nx.average_clustering(updated_graph)

        update_graph_list.append(updated_graph)
        avg_clustering_list.append(avg_clustering)

    """vertex with most effect (which reduces avg clustering)
        is opt_vert"""
    opt_vert = np.argmin(avg_clustering_list)
    op_graph = update_graph_list[opt_vert]

    if nx.average_clustering(inp_graph) <= nx.average_clustering(op_graph):
        op_graph = inp_graph
        flagg = 0

    return op_graph, vert_list[opt_vert], flagg

def apply_new_metric(inp_graph):
    update_graph_list = []
    avg_clustering_list = []
    flagg = 1
    vert_list = new_metric_nodes(inp_graph)

    for i, vert in enumerate(vert_list):
        updated_graph = local_complementation(inp_graph, vert)
        avg_clustering = nx.average_clustering(updated_graph)

        update_graph_list.append(updated_graph)
        avg_clustering_list.append(avg_clustering)

    """vertex with most effect (which reduces avg clustering)
        is opt_vert"""
    opt_vert = np.argmin(avg_clustering_list)
    op_graph = update_graph_list[opt_vert]

    if nx.average_clustering(inp_graph) <= nx.average_clustering(op_graph):
        op_graph = inp_graph
        flagg = 0

    return op_graph, vert_list[opt_vert], flagg

def minimisation_clustering(inp_graph):
    num_edges_list = []
    flagg = 1
    i=0
    while flagg == 1:
        num_edges_list.append(inp_graph.number_of_edges())
        inp_graph, vert, flagg = apply_lc_clustering(inp_graph)
        i += 1
        if i >= 100:
            flagg = 0
    return np.min(num_edges_list)

def minimisation_new_metric(inp_graph):
    num_edges_list = []
    flagg = 1
    i = 0
    while flagg == 1:
        num_edges_list.append(inp_graph.number_of_edges())
        inp_graph, vert, flagg = apply_new_metric(inp_graph)
        i += 1
        if i >= 100:
            flagg = 0
    return np.min(num_edges_list)

def minimisation_triangle(inp_graph):
    num_edges_list = []
    flagg = 1
    i = 0
    while flagg == 1:
        num_edges_list.append(inp_graph.number_of_edges())
        inp_graph, vert, flagg = apply_lc_triangles(inp_graph)
        i += 1
        if i >= 100:
            flagg = 0
    return np.min(num_edges_list)

if __name__ == "__main__":
    err = []
    cl_data = []
    nm_data = []
    tr_data = []
    edge_data = []
    G_list = []
    cl_avg = []
    nm_avg = []
    tr_avg = []
    edge_avg = []
    x = []
    n = 80
    p = 0.6
    for j in range(40):
        print(j)
        p = 0.025*(j+1)
        x.append(p)
        edge_list = []
        nm_list = []
        cl_list = []
        tr_list = []
        cl_graph_list = []
        tr_graph_list = []
        nm_graph_list = []
        #it = 500*(j+1)

        for i in range(3000):
                
            G = nx.fast_gnp_random_graph(n, p)
            edge_list.append(G.number_of_edges())
            G_list.append(G)
            min_cl_edge = minimisation_clustering(G)
            min_nm_edge = minimisation_new_metric(G)
            min_tr_edge = minimisation_triangle(G)

            cl_list.append(min_cl_edge)
            nm_list.append(min_nm_edge)
            tr_list.append(min_tr_edge)

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
        cl_avg.append(np.mean(cl_list))
        nm_avg.append(np.mean(nm_list))
        tr_avg.append(np.mean(tr_list))
    print(len(err), "wrong moves")
    
      
    data_dict = {
        "prob_list": x,
        "edge_data": edge_data,
        "cl_data": cl_data,
        "nm_data": nm_data,
        "tr_data": tr_data,  
        "G_list": G_list,
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
    
    
    with open(__file__) as f:
        data = f.read()
        f.close()
        
    plt.figure()
    plt.grid()
    plt.plot(x, cl_avg, '-o', x, nm_avg, '-o', x, tr_avg, '-o', x, edge_avg, '-o')
    #plt.plot(x, cl_avg, x, nm_avg, x, tr_avg, x, edge_avg)
    plt.ylabel('Average number of edges')
    plt.xlabel('Probability')
    plt.legend(["Clustering", "New metric", "# triangles", "edge"])
    plt.savefig(plots_directory + time_str + "_n="+str(n)+"_performance" + ".png", dpi=1000, format="png", bbox_inches = 'tight')
    """
    metadata_dict = {
        "num of vertex": n,
        "prob": np.max(alpha_list),
        "alpha_min": np.min(alpha_list),
        "prob_in_state_max": np.max(prob_in_state_list),
        "prob_in_state_min": np.min(prob_in_state_list),
        "depol_channel": "x"
        }
    
    with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
        f.write(str(metadata_dict))
        f.close()
    """