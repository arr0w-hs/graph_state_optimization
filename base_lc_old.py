#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:08:32 2024

@author: hsharma4

base functions for applying local complementation
with greedy appraoch
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
    #triangles_val = list(triangles_dict.values())
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
    #print(nx.algebraic_connectivity(op_graph))
    return op_graph, vert_list[opt_vert], flagg

def minimisation_clustering(inp_graph):
    num_edges_list = []
    flagg = 1
    flagg_count=0
    while flagg == 1:
        num_edges_list.append(inp_graph.number_of_edges())
        inp_graph, vert, flagg = apply_lc_clustering(inp_graph)
        flagg_count += 1
        if flagg_count >= 30:
            print("cl hit max")
            flagg = 0
    return np.min(num_edges_list), inp_graph

def minimisation_new_metric(inp_graph):
    num_edges_list = []
    flagg = 1
    flagg_count = 0
    while flagg == 1:
        num_edges_list.append(inp_graph.number_of_edges())
        inp_graph, vert, flagg = apply_new_metric(inp_graph)
        flagg_count += 1
        if flagg_count >= 30:
            print("nm hit max")
            flagg = 0
    return np.min(num_edges_list), inp_graph

def minimisation_triangle(inp_graph):
    num_edges_list = []
    flagg = 1
    flagg_count = 0
    while flagg == 1:
        num_edges_list.append(inp_graph.number_of_edges())
        inp_graph, vert, flagg = apply_lc_triangles(inp_graph)
        flagg_count += 1
        if flagg_count >= 30:
            print("tr hit max")
            flagg = 0
    return np.min(num_edges_list), inp_graph



if __name__ == "__main__":
    print(1)