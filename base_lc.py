#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:10:24 2024

@author: hsharma4

base functions for applying local complementation
with greedy appraoch
"""
import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
plt.rcParams.update({'font.size': 12})

class Greedy:
    """Class for implementing greedy algorithm with different heuristics"""

    def __init__(self, inp_graph):
        self.inp_graph = inp_graph


    def local_complementation(self, graph, vert):
        """function for local complementation at vertex 'vert'"""

        adj_mat = nx.to_numpy_array(graph)
        row = adj_mat[vert]
        adj_mat = adj_mat + np.outer(row, row)
        np.fill_diagonal(adj_mat, 0)
        adj_mat = adj_mat%2

        return nx.from_numpy_array(adj_mat)


    def new_metric_nodes(self):
        """metric with multiplication of clustering with degree"""

        degree_list = [val for (node, val) in self.inp_graph.degree()]

        clustering_dict = nx.clustering(self.inp_graph)
        clustering_val = list(clustering_dict.values())

        new_metric_val = np.multiply(degree_list, clustering_val)
        vertices = np.flatnonzero(new_metric_val == np.max(new_metric_val))
        if len(vertices) == 0:
            vertices = np.flatnonzero(clustering_val == np.max(clustering_val))

        return vertices


    def apply_new_metric(self, graph):
        """apply local comp at vertex chosen based on metric"""

        update_graph_list = []
        avg_clustering_list = []
        flagg = 1
        vert_list = self.new_metric_nodes()

        in_graph = graph

        for i, vert in enumerate(vert_list):
            updated_graph = self.local_complementation(in_graph, vert)
            avg_clustering = nx.average_clustering(updated_graph)

            update_graph_list.append(updated_graph)
            avg_clustering_list.append(avg_clustering)

        """vertex with most effect (which reduces avg clustering)
            is opt_vert"""
        opt_vert = np.argmin(avg_clustering_list)
        op_graph = update_graph_list[opt_vert]

        if nx.average_clustering(in_graph) <= nx.average_clustering(op_graph):
            op_graph = in_graph
            flagg = 0
        #print(nx.algebraic_connectivity(op_graph))

        return op_graph, vert_list[opt_vert], flagg


    def greedy_minimisation(self):
        """minimise using greedy approach"""

        num_edges_list = []
        flagg = 1
        flagg_count = 0
        graph = self.inp_graph
        while flagg == 1:
            num_edges_list.append(graph.number_of_edges())
            #graph, vert, flagg = self.apply_new_metric(graph)
            output  = self.apply_new_metric(graph)
            flagg = output[2]
            flagg_count += 1
            if flagg_count >= 30:
                print("nm hit max")
                flagg = 0

        return np.min(num_edges_list), output[0]




if __name__ == "__main__":

    G = nx.fast_gnp_random_graph(10, 0.5)
    print(G.number_of_edges())

    opt = Greedy(G)
    edges, opg = opt.greedy_minimisation()
    print(edges)
