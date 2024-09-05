#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:25:33 2024

@author: hsharma4

generating ER graphs with probabilities going form 0 to 1, 
for given set of vertices
"""



import sys
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import networkx as nx

dir_name = os.path.dirname(__file__)
os.chdir(dir_name)
sys.path.append('..')

from edge_minimisation.gen_bounded_deg import check_is_iso

def create_er(sample_size, n, num_points):
    """
    function for creating a list of er graphs from p=0 to 1

    inputs:
        sample size: the number of graphs requested
        n: is the number of vertices in the graphs
        num_points: is the number of points in the probability from 0 to 1

    outputs: a pickle file with a dictionary with graphs and a corresponding
            list of probabilites
    """

    g_data = {}

    p_list = np.linspace(0, 1, num_points, endpoint=False)
    p_list = p_list + 1/num_points
    p_list = p_list[:-1]
    print(p_list)
    for p_ele in (p_list):

        if p_ele == 1:
            continue

        g_list = []

        flag = 1
        stop = 0

        while flag <= sample_size and stop < 10000:
            stop += 1

            G = nx.erdos_renyi_graph(n, p_ele)
            if not check_is_iso(g_list, G):

                flag += 1
                g_list.append(G)

        print(len(g_list), "len glist", p_ele)
        g_data[str(p_ele)] = g_list

    print(g_data.keys())
    graph_dict = {
        "g_data": g_data,
        "p_list": p_list,
        #"gout_data": gout_data,
        }

    metadata_dict = {
        "p_list": p_list,
        "sample size": sample_size,
        "n": n,
        }

    ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
    date_str = str(ts.date())

    time_str = ts.time()
    time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
    print(time_str)
    #data_directory = os.path.join(dir_name+"/data", date_str+"_sa_ilp/")
    graph_directory= os.path.join(dir_name+"/er_graphs/")
    #plots_directory = os.path.join(dir_name+"/plots", date_str+"_sa_ilp/")

    graph_folder = Path(graph_directory)
    if not graph_folder.exists():
        os.mkdir(graph_directory)

    with open(graph_directory+ str(n) +"_"+ str(sample_size) +"_"+ str(num_points)
              + "_"+ date_str + "_" + time_str +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(graph_dict, f)

    with open(graph_directory+ str(n) +"_"+ str(sample_size) +"_"+ str(num_points)
              + "_" + time_str +'_metadata.txt', mode="w") as f:
        f.write(str(metadata_dict))
        f.close()
    print(str(n) +"_"+ str(sample_size) +"_"+ str(num_points)
              + "_"+ date_str + "_" + time_str +'.pkl')
    return g_data

if __name__ == "__main__":

    gdata = create_er(100, 6, 15)
