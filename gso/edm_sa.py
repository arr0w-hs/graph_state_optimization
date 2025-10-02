#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:14:37 2024

@author: hsharma4

code for the class to implement simulated annealing
"""
import sys
import os
import numpy as np
import networkx as nx
dir_name = os.path.dirname(__file__)
os.chdir(dir_name)
sys.path.append('..')

#from gso.gsc.is_lc_equiv import are_lc_equiv
import time
#import itertools as it
import matplotlib.pyplot as plt
import pandas as pd


class EDM_SimAnnealing:
    """Class for simulated annealing"""
    def  __init__(self, inp_graph, k_max, initial_temp):
        self.inp_graph = inp_graph
        self.k_max = k_max
        self.initial_temp = initial_temp
        #self.transition_cutoff = transition_cutoff


    def local_complementation(self, in_graph, vert):
        """function for local complementation at vertex 'vert'"""
        edge_list = list(in_graph.edges())
        
        dim = in_graph.number_of_nodes()
        adj_mat = np.zeros([dim,dim])

        for ele in edge_list:
            adj_mat[ele[0],ele[1]] = 1
            adj_mat[ele[1],ele[0]] = 1
        row = adj_mat[vert]

        adj_mat = adj_mat + np.outer(row, row)

        #np.fill_diagonal(adj_mat, 0)
        dim = np.shape(adj_mat)[0]
        for i in range(dim):
            adj_mat[i,i] = 0

        adj_mat = adj_mat%2
        out_graph = nx.from_numpy_array(adj_mat)

        return out_graph


    def vertex_choice(self, in_graph, transition_cutoff, vertex_met):
        """func for choosing vertex at each point of optimisation"""

        degree_list = [val for (node, val) in in_graph.degree()]
        clustering_dict = nx.clustering(in_graph)
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
        assert self.k_max !=0
        cutoff_arg = np.ceil((transition_cutoff/self.k_max)*len(unique_elements))

        if cutoff_arg < len(unique_elements):
            chosen_arg = np.random.randint(cutoff_arg, len(unique_elements))

        else:
            chosen_arg = len(unique_elements) - 1

        vertices = [index for index in np.flatnonzero(new_metric_val)
                    if new_metric_val[index] == unique_elements[chosen_arg]]

        if len(vertices) == 0:
            vertices = np.flatnonzero(clustering_val == np.max(clustering_val))


        if vertex_met:
            vertex = np.random.choice(vertices)
        else:
            vertex = np.random.choice(nx.nodes(in_graph))

        return vertex


    def energy_func(self, in_graph, metric):
        """function for calculating energy, num edges in this case"""

        if metric == "number of edges":
            energy = in_graph.number_of_edges()
        elif metric == "connectivity":
            energy = nx.algebraic_connectivity(in_graph, method='lanczos')

        return energy


    def simulated_annealing(self, metric, vertex_met = True, lc_test = False):
        """func for implementing simulated annealing
        g_best is the output graph
        x_best is the list of palces where we act with local comp"""

        temp = self.initial_temp
        transition_cutoff = 1
        g_best = self.inp_graph
        graph = self.inp_graph
        y = self.energy_func(graph, metric)
        y_best = y
        edge_list_best = []
        x_list = []

        edge_list_best.append(y_best)
        for k in range(self.k_max):
            x_new = self.vertex_choice(graph, transition_cutoff, vertex_met)

            #g_new = self.local_complementation(g, x_new)
            g_new = self.local_complementation(graph, x_new)
            y_new = self.energy_func(g_new, metric)

            if y_new - y <= 0 or np.random.uniform(0,1,1) < np.exp(-1*(y_new - y)/temp):
                y = y_new
                graph = g_new
                x_list.append(x_new)
                edge_list_best.append(y_new)

            if y_new < y_best:
                g_best = g_new
                y_best = y_new

            temp = self.initial_temp*np.log(2)/(np.log(k+2))
            #temp = self.initial_temp/(k+2)
            transition_cutoff = k+1


        # if nx.is_connected(self.inp_graph) and lc_test:
        #     output = are_lc_equiv(self.inp_graph, g_best)
        #     if not output[0]:
        #         print(output[0], "sa gave a non lc-equivalent output")
        # print(len(x_list), len(edge_list_best))f

        edge_list_best = edge_list_best[:np.argmin(edge_list_best)+1]
        x_list = x_list[:np.argmin(edge_list_best)]
        x_list = x_list[::-1]

        return g_best, edge_list_best, x_list

if __name__ == "__main__":

    t_list = []
    in_list = []
    klist = []
    out_list = []
    templist = []
    out_dict = {}
    for k in range(100):
        print(k)

        G = nx.erdos_renyi_graph(100, 0.6)

        for i in range(0, 27):
            # G = rgs_graph(10)#, 5, True)
            kmax = 1000*i+50
            for j in range(1):
                temp = 50

                sa1 = EDM_SimAnnealing(G, kmax, temp)
                t = time.time()
                gout, _, _ = sa1.simulated_annealing("number of edges")
                t_list.append(time.time()-t)

                in_list.append(G.number_of_edges())#, "edges in original")
                out_list.append(gout.number_of_edges())#
                klist.append(kmax)
                # templist.append(temp)

    out_dict["kmax"] = klist
    # out_dict["temp"] = templist
    out_dict["edge"] = in_list
    out_dict["sa_edge"] = out_list
    out_dict["time"] = t_list

    # fs = 15
    # plt.figure()
    # plt.plot(klist, out_list, label= "Initial edges")
    # plt.plot(pl, ol, label= r"Using $M_{v}$")
    # plt.plot(pl, ol1, label= "Random vertex choice")
    # plt.ylabel('Number of edges', fontsize=fs)
    # #plt.xlabel('Raw state infidelity')
    # # plt.xlabel('Coherent error', fontsize=fs)
    # plt.xlabel('Probability', fontsize=fs)
    # plt.xticks(fontsize=fs)
    # plt.yticks(fontsize = fs)
    # plt.legend(fontsize = fs,
    #            handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
    # plt.grid()
    # # plt.savefig(dir_name +'/plots' + "metric_run2" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
    
    # plt.show()
    
    out_df = pd.DataFrame(out_dict)
    wide = out_df.groupby(['kmax'],as_index=False).mean()
    
    # print(out_df)
    print(wide)

