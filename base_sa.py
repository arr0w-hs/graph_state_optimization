#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:14:37 2024

@author: hsharma4

code for the class to implement simulated annealing
"""
from pathlib import Path

import sys
import os


import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
from base_lc import Greedy as greedy


plt.rcParams.update({'font.size': 12})


class SimAnnealing:
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


if __name__ == "__main__":

    #print(lauda)
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

    sa_min = []

    n = 10
    x = []
    max_k = 10*n
    temp_initial = 100
    """sample size is the number of graph sampled for each (n,p)"""
    sample_size = 1000

    for j in range(1):

        print(j)
        p = 0.05*(j+1)
        #p = 0.2
        x.append(j)

        edge_list = []
        nm_list = []
        sa_list = []
        gout_list = []
        g_list = []
        
        #print(min_nm_edge)

        #plt.figure()
        #nx.draw_networkx(G)
        #plt.draw()
        #plt.show(block = False)

        for i in range(1):
            #G = nx.fast_gnp_random_graph(n, p)
            G = nx.Graph()
            G.add_edges_from([(0, 1), (1, 6), (6, 2), (2, 3), (3, 5), (5, 0), (0, 7), (7, 1), (7, 3), (7, 2), (7, 4), (4, 6), (4, 5)])

            nm1 = greedy(G)
            min_nm_edge, min_nm_graph = nm1.greedy_minimisation()
            
            sa1 = SimAnnealing(G, max_k, temp_initial)
            g_out, y_list, ui_list = sa1.simulated_annealing("number of edges")
            #g_out, y_list, ui_list, ac = sa1.sa("connectivity")
            #print(np.min(ui_list))
            """if g_out.number_of_edges() < G.number_of_edges():
                print((g_out.number_of_edges()), G.number_of_edges())
                print(min_nm_edge)
                gout_list.append(g_out)
                g_list.append(G)
            """    
            edge_list.append(G.number_of_edges())
            sa_list.append((g_out.number_of_edges()))
            nm_list.append(min_nm_edge)

            #xx = np.linspace(0, len(y_list), len(y_list))
            #plt.figure()
            #plt.grid()
            #plt.plot(xx, y_list, xx, ui_list)
            #plt.show(block = False)


        edge_data.append(edge_list)
        nm_data.append(nm_list)
        sa_data.append(sa_list)
        g_data.append(g_list)

        edge_avg.append(np.mean(edge_list))
        nm_avg.append(np.mean(nm_list))
        sa_avg.append(np.mean(sa_list))
        #g_data.append(g_list)#

        sa_min.append(np.min(sa_list))
    print(sa_min)
    #print(nm_avg, sa_avg)
    print(np.min(nm_data), np.min(sa_data))
    #plt.figure()
    #plt.grid()
    #plt.plot(x, num_edges_list, '-o', x, nm_list, '-o', x, sa_list, '-o')
    #plt.plot(x, edge_avg, '-o',  x, nm_avg, '-o',  x, sa_min, '-o')
    #plt.plot(x, edge_avg, x, nm_avg, x, sa_min)
    #plt.title("n="+str(n))
    #plt.plot(x, cl_avg, x, nm_avg, x, tr_avg, x, edge_avg)
    #plt.ylabel('Number of edges')
    #plt.xlabel('Graph')
    #plt.legend(["Initial edges", "New metric", "Simulated annealing"])
    #plt.savefig(plots_directory+time_str+"_n="+str(n)+"_performance"+".png",
    #           dpi=1000, format="png", bbox_inches = 'tight')

    #print(lauda)

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
    plt.plot(x, ui_list)
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

    with open(graph_directory+ str(sample_size) 
              + "_"+ date_str + "_" + time_str +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(graph_dict, f)

    with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
        f.write(str(metadata_dict))
        f.close()

    #x = np.linspace(0, itera, itera)
    plt.figure()
    plt.grid()
    #plt.plot(x, num_edges_list, '-o', x, nm_list, '-o', x, sa_list, '-o')
    plt.plot(x, edge_avg, x, nm_avg,  x, sa_min)
    plt.title("n="+str(n))
    #plt.plot(x, cl_avg, x, nm_avg, x, tr_avg, x, edge_avg)
    plt.ylabel('Number of edges')
    plt.xlabel('Graph')
    plt.legend(["Initial edges", "New metric", "Simulated annealing"])
    plt.savefig(plots_directory+time_str+"_n="+str(n)+"_performance" + ".png",
                dpi=1000, format="png", bbox_inches = 'tight')
