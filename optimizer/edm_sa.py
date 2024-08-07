#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:14:37 2024

@author: hsharma4

code for the class to implement simulated annealing
"""
import sys
import os
dir_name = os.path.dirname(__file__)
os.chdir(dir_name)
sys.path.append('..')


import numpy as np
import networkx as nx


class EDM_SimAnnealing:
    """Class for simulated annealing"""
    def  __init__(self, inp_graph, k_max, initial_temp):
        self.inp_graph = inp_graph
        self.k_max = k_max
        self.initial_temp = initial_temp
        #self.transition_cutoff = transition_cutoff


    def local_complementation(self, graph, vert):
        """function for local complementation at vertex 'vert'"""

        adj_mat = nx.to_numpy_array(graph)
        row = adj_mat[vert]
        adj_mat = adj_mat + np.outer(row, row)
        np.fill_diagonal(adj_mat, 0)
        adj_mat = adj_mat%2

        return nx.from_numpy_array(adj_mat)


    def vertex_choice(self, graph, transition_cutoff):
        """func for choosing vertex at each point of optimisation"""

        degree_list = [val for (node, val) in graph.degree()]
        clustering_dict = nx.clustering(graph)
        clustering_val = list(clustering_dict.values())
        new_metric_val = np.multiply(degree_list, clustering_val)

        if np.sum(new_metric_val) != 0:
            new_metric_val = new_metric_val/np.sum(new_metric_val)

        """
        find all the unique elemetns in the array
        and randomly choose an element form the vertices which are
        larger than the value of cutoff_arg at that point
        """
        unique_elements = np.unique(new_metric_val)
        #cutoff_arg = np.ceil((1-1/transition_cutoff)*len(unique_elements))
        if self.k_max != 0:
            cutoff_arg = np.ceil((transition_cutoff/self.k_max)*len(unique_elements))
        else:
            raise Exception("k_max is 0")

        #cutoff_arg = 0
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


    def energy_func(self, graph, metric):
        """function for calculating energy, num edges in this case"""

        if metric == "number of edges":
            energy = graph.number_of_edges()
        elif metric == "connectivity":
            energy = nx.algebraic_connectivity(graph, method='lanczos')

        return energy


    def simulated_annealing(self, metric):
        """func for implementing simulated annealing"""

        temp = self.initial_temp
        transition_cutoff = 1
        g_best = self.inp_graph
        graph = self.inp_graph
        y = self.energy_func(graph, metric)
        y_best = y

        edge_list_sa = []
        best_list_sa = []
        for k in range(self.k_max):
            x_new = self.vertex_choice(graph, transition_cutoff)
            #g_new = self.local_complementation(g, x_new)
            g_new = self.local_complementation(graph, x_new)
            y_new = self.energy_func(g_new, metric)

            dy = y_new - y

            if dy <= 0 or np.random.uniform(0,1,1) < np.exp(-1*dy/temp):
                y = y_new
                graph = g_new

            if y_new < y_best:
                g_best = g_new
                y_best = y_new

                #plt.figure()
                #nx.draw_networkx(g_best)
                #plt.draw()
                #plt.show(block = False)

            #plt.figure()
            #nx.draw_networkx(graph)
            #plt.draw()
            #plt.show(block = False)

            temp = self.initial_temp*np.log(2)/(np.log(k+2))
            #temp = self.initial_temp/(k+2)
            transition_cutoff = k+1
            edge_list_sa.append(y)
            best_list_sa.append(y_best)

            #ac.append(nx.algebraic_connectivity(g_best, method='lanczos'))

        return g_best, edge_list_sa, best_list_sa#, ac
