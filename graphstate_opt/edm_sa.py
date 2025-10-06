#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:14:37 2024

@author: hsharma4

code to implement simulated annealing edge minimisation
"""
import numpy as np
import networkx as nx
import time
import pandas as pd
from typing import Optional

def local_complementation(in_graph, vert):
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


def vertex_choice(in_graph, transition_cutoff, k_max, vertex_met):
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
    assert k_max !=0
    cutoff_arg = np.ceil((transition_cutoff/k_max)*len(unique_elements))

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


def energy_func(in_graph, metric):
    """function for calculating energy, num edges in this case"""

    if metric == "edge_count":
        energy = in_graph.number_of_edges()
    elif metric == "connectivity":
        energy = nx.algebraic_connectivity(in_graph, method='lanczos')

    return energy


def edm_sa(in_graph: nx.Graph, k_max: int, initial_temp: float,
            metric:Optional[str]= "edge_count", 
            vertex_met: bool = True):
    """
    Find an approximate Minimum Edge Representative (MER) using simulated annealing.

    The algorithm applies simulated annealing (SA) for a fixed number of iterations,
    performing local complementations to reduce the chosen metric (by default, the
    edge count). Optionally, vertices are selected using a heuristic (clustering
    coefficient) or uniformly at random.

    Parameters
    ----------
    in_graph : nx.Graph
        Input graph whose MER is to be approximated.
    k_max : int
        Maximum number of simulated annealing iterations.
    temp : float
        Initial temperature for simulated annealing.
    metric : str, optional
        Metric to minimize. Defaults to "edge_count".
        Use "connectivity" to minimize algebraic connectivity instead.
    vertex_met : bool, optional
        If True (default), choose the vertex for local complementation using a
        heuristic (e.g., clustering coefficient). If False, choose vertices
        uniformly at random.

    Returns
    -------
    g_best : nx.Graph
        The best (lowest-metric) graph encountered during SA (approximate MER).
    edge_list_best : list of int
        History of the best metric value after each iteration.
        (If `metric="edge_count"`, these are best-so-far edge counts.)
    x_list : list of int
        The sequence of vertex indices (or IDs) where local complementation
        was applied to move from the input graph toward the MER.

    Notes
    -----
    - Local complementation is applied as the move operator.
    - If `metric="connectivity"`, algebraic connectivity is evaluated at each step.
      This can be more expensive than edge counting.
    """

    temp = initial_temp
    transition_cutoff = 1
    g_best = in_graph
    graph = in_graph
    y = energy_func(graph, metric)
    y_best = y
    edge_list_best = []
    x_list = []

    edge_list_best.append(y_best)
    for k in range(k_max):
        x_new = vertex_choice(graph, transition_cutoff, k_max, vertex_met)

        #g_new = local_complementation(g, x_new)
        g_new = local_complementation(graph, x_new)
        y_new = energy_func(g_new, metric)

        if y_new - y <= 0 or np.random.uniform(0,1,1) < np.exp(-1*(y_new - y)/temp):
            y = y_new
            graph = g_new
            x_list.append(x_new)
            edge_list_best.append(y_new)

        if y_new < y_best:
            g_best = g_new
            y_best = y_new

        temp = initial_temp*np.log(2)/(np.log(k+2))
        #temp = initial_temp/(k+2)
        transition_cutoff = k+1

    edge_list_best = edge_list_best[:np.argmin(edge_list_best)+1]
    x_list = x_list[:np.argmin(edge_list_best)]
    x_list = x_list[::-1]

    return g_best, edge_list_best, x_list

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t_list = []
    in_list = []
    klist = []
    out_list = []
    templist = []
    out_dict = {}
    for k in range(10):
        print(k)

        G = nx.erdos_renyi_graph(10, 0.6)

        for i in range(0, 5):
            # G = rgs_graph(10)#, 5, True)
            kmax = 1000*i+50
            for j in range(1):
                temp = 50
                t = time.time()
                gout, _, _ =  edm_sa(G, kmax, temp)
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

    fs = 15
    plt.figure()
    plt.plot(klist, out_list, label= "Initial edges")
    # plt.plot(pl, out_list, label= r"Using $M_{v}$")
    # plt.plot(pl, out_list, label= "Random vertex choice")
    plt.ylabel('Number of edges', fontsize=fs)
    plt.xlabel('Probability', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize = fs)
    plt.legend(fontsize = fs,
               handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
    plt.grid()
    # plt.savefig(dir_name +'/plots' + "metric_run2" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
    
    plt.show()
    
    out_df = pd.DataFrame(out_dict)
    wide = out_df.groupby(['kmax'],as_index=False).mean()
    
    # print(out_df)
    print(wide)