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


def check_is_iso(glist, graph):
    out = None
    
    if len(glist) > 0:
        for ele in glist:
            if nx.is_isomorphic(graph, ele):
                out = True
                return True
            else:
                out = False
    else:
        return False
    
    return out

def create_bounded_deg(sample_size, d_max):
    g_data = {}
    n_list = []
    for i in range(1):
        
        n = 6+i
        g_list = []
        n_list.append(n)
        
        flag = 1
        stop = 0
    
        while flag <= sample_size and stop < 2000000:
            stop += 1
            deg_seq = np.random.randint(1, high = d_max+1, size = n)
            if nx.is_graphical(deg_seq):
                G = nx.havel_hakimi_graph(deg_seq)
                if nx.is_connected(G):
                    
                    if not check_is_iso(g_list, G):
                        
                        #print(G.number_of_edges())
                        #plt.figure()
                        #nx.draw_networkx(G)
                        #plt.draw()
                        #plt.show(block = False)
                        
                        flag += 1
                        
                        g_list.append(G)
                
        print(stop)        
        print(len(g_list), "len glist", n)
        g_data[str(n)] = g_list

    print(g_data.keys())
    #print(afaalkjfak)
    print(type(g_data))
    graph_dict = {
        "g_data": g_data,
        "d_max": d_max,
        #"gout_data": gout_data,
        }

    metadata_dict = {
        "n_list": n_list,
        "sample size": sample_size,
        "d_max": d_max,
        }

    ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
    date_str = str(ts.date())

    time_str = ts.time()
    time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
    print(time_str)
    #data_directory = os.path.join(dir_name+"/data", date_str+"_sa_ilp/")
    graph_directory= os.path.join(dir_name+"/bounded_deg_graphs/")
    #plots_directory = os.path.join(dir_name+"/plots", date_str+"_sa_ilp/")

    graph_folder = Path(graph_directory)
    if not graph_folder.exists():
        os.mkdir(graph_directory)

    with open(graph_directory+ str(sample_size)
              + "_"+ date_str + "_" + time_str +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(graph_dict, f)

    with open(graph_directory + time_str +'_metadata.txt', mode="w") as f:
        f.write(str(metadata_dict))
        f.close()
    
    return g_data

if __name__ == "__main__":
    
    gdata = create_bounded_deg(100, 6)
    
    #print(check_is_iso(gdata[5], gdata[5][26]))
