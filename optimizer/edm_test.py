#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on Feb 26 2025

@author: hsharma4

code for testing M(v)
"""
import sys
import os
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import pandas as pd

# from . import edm_sa as edm
from optimizer.gsc.is_lc_equiv import are_lc_equiv
from edm_sa import EDM_SimAnnealing as sa


if __name__ == "__main__":

    
    ol1 = []
    ol = []
    il = []
    pl = []
    for j in range(10):
        t_list = []
        in_list = []
        out_list1 = []
        out_list = []

        p = 0.1+j*0.1
        print(p)
        pl.append(p)
        for i in range(5):
            # G = rgs_graph(10)#, 5, True)


            G = nx.erdos_renyi_graph(15, p)

            # plt.figure()
            # nx.draw_networkx(G)
            # plt.draw()
            # plt.show(block = False)
            sa1 = sa(G, 100, 100)
            t = time.time()
            gout, _, _ = sa1.simulated_annealing("number of edges")
            t_list.append(time.time()-t)
            gout1, _, _ = sa1.simulated_annealing("number of edges", vertex_met=False)
            # print(time.time()-t)
            in_list.append(G.number_of_edges())#, "edges in original")
            out_list.append(gout.number_of_edges())#
            out_list1.append(gout1.number_of_edges())
            # plt.figure()
            # nx.draw_networkx(gout)
            # plt.draw()
            # plt.show(block = False)
        # print(out_list)
        # print(in_list)
        il.append(np.average(in_list))
        ol.append(np.average(out_list))
        ol1.append(np.average(out_list1))#, "ol1")
    # print(np.average(t_list))

fs = 15
plt.figure()
plt.plot(pl, il, label= "Initial edges")
plt.plot(pl, ol, label= r"Using $M_{v}$")
plt.plot(pl, ol1, label= "Random vertex choice")
plt.ylabel('Number of edges', fontsize=fs)
#plt.xlabel('Raw state infidelity')
# plt.xlabel('Coherent error', fontsize=fs)
plt.xlabel('Probability', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize = fs)
plt.legend(fontsize = fs,
           handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
plt.grid()
# plt.savefig(dir_name +'/plots' + "metric_run2" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')

plt.show()
# fs = 15
# plt.figure()
# plt.plot(pl, il, label= "Initial edges")
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

