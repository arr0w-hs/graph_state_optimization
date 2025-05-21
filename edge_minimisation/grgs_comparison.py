#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 23:22:38 2025

@author: hsharma4
comparison of fusion creation of G+L and mer(G)+L
using optgraphstate
"""

import sys
import os
sys.path.append('..')

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

from grgs_draw import add_leaves
from optimizer.edm_sa import EDM_SimAnnealing as sa
import optgraphstate as ogs


out_dict = {}
prob_list = []
edge_list = []
sa_list = []
sa_wl_list = []
gout_list = []
g_list = []
ghz_list = []
mer_ghz_list = []
mer_wl_ghz_list = []
fus_list = []
mer_fus_list = []
mer_wl_fus_list = []

G=nx.Graph()
edge_list1 = [(0, 10), (0, 13), (0, 14), (0, 16), (0, 17), (1, 9),
              (1, 11), (1, 12), (1, 13), (1, 14), (1, 16), (1, 17),
              (2, 10), (2, 13), (2, 14), (2, 16), (2, 17), (3, 10),
              (3, 12), (3, 13), (3, 14), (3, 15), (4, 9), (4, 10),
              (4, 11), (4, 12), (5, 9), (5, 10), (5, 11), (5, 12),
              (6, 12), (6, 15), (6, 16), (6, 17), (7, 9), (7, 10),
              (7, 11), (7, 15), (7, 16), (7, 17), (8, 9), (8, 10),
              (8, 11), (8, 15), (8, 16), (8, 17)]
G.add_edges_from(edge_list1)


pos = nx.bipartite_layout(G, list(range(9)))

sa1 = sa(G, 10000, 100)
gout, _, _ = sa1.simulated_annealing("number of edges")

n = G.number_of_nodes()
plt.figure()
nx.draw_networkx(G)
plt.draw()
plt.show(block=False)


for j in range(5):
    p = 0.1+j*0.1
    print(p)
    for i in range(1):

        G_leaves = add_leaves(G)
        gs = ogs.GraphState(graph = G_leaves)
        res = gs.simulate(10000, p_succ = p, unravel_bcs_first=True, pbar=False, verbose = False)
        edge_list.append(G_leaves.number_of_edges())
        ghz_list.append(res['best_overhead'])
        fus_list.append(res['best_num_fusions'])

        # node_names, fusions, final_qubits = gs.get_instructions()
        # print("fusions")
        # for num, val in fusions.items():
        #     print(num, val)
        # print(node_names)
        # # print(fusions)
        # print(final_qubits)


        sa1 = sa(G, 10000, 100)
        gout, _, _ = sa1.simulated_annealing("number of edges")
        gout = add_leaves(gout)
        gs = ogs.GraphState(graph = gout)
        res = gs.simulate(10000, p_succ = p, unravel_bcs_first=True, pbar=False, verbose = False)

        sa_list.append(gout.number_of_edges())
        mer_ghz_list.append(res['best_overhead'])
        mer_fus_list.append(res['best_num_fusions'])

        # node_names, fusions, final_qubits = gs.get_instructions()
        # print("fusions")
        # for num, val in fusions.items():
        #     print(num, val)
        # print(node_names)
        # # print(fusions)
        # print(final_qubits)

        prob_list.append(p)
        out_dict["prob_val"] = prob_list
        out_dict["G_edge"] = edge_list
        out_dict["sa_edge"] = sa_list
        out_dict["ghz_list"] = ghz_list
        out_dict["mer_ghz_list"] = mer_ghz_list
        out_dict["fus_list"] = fus_list
        out_dict["mer_fus_list"] = mer_fus_list

out_df = pd.DataFrame(out_dict)
wide = out_df.groupby(['prob_val'],as_index=False).mean()
max_df = out_df.groupby(['prob_val'],as_index=False).max()
min_df = out_df.groupby(['prob_val'],as_index=False).min()
fs = 15

plt.figure()
plt.plot(wide['prob_val'], wide['ghz_list'], label = "G+L", marker = 's')
plt.plot(wide['prob_val'], wide['mer_ghz_list'], label = "H+L", marker = 'v')
plt.xlabel("Success probability of fusions", fontsize=fs, labelpad=1)
plt.ylabel("Average GHZ states required", fontsize=fs)
plt.yscale("log")
plt.xticks(fontsize=fs)
plt.xticks(np.arange(0.1, 0.6, step=0.1), fontsize=fs)
plt.yticks(fontsize=15)
plt.grid()
# plt.tick_params(direction='out', length=6, width=200,
#                 grid_alpha=0.5 , pad = 1)
plt.legend(fontsize = 13, title="Method", title_fontsize=15,
            handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
# plt.savefig(dir_name + "/ghz_plot" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
# plt.savefig(dir_name + "/ghz_1_high" + ".svg", dpi=1000, format="svg", bbox_inches = 'tight')
plt.show()


plt.figure()
plt.plot(wide['prob_val'], wide['fus_list'], label = "G+L", marker = 's')
plt.plot(wide['prob_val'], wide['mer_fus_list'], label = "H+L", marker = 'v')
plt.xlabel("Success probability of fusions", fontsize=fs, labelpad=1)
plt.ylabel("Average fusions required", fontsize=fs)
plt.xticks(np.arange(0.1, 0.6, step=0.1), fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=15)
plt.yscale("log")
plt.grid()
# plt.tick_params(direction='out', length=6, width=200,
#                 grid_alpha=0.5 , pad = 1)
plt.legend(fontsize = 13, title="Method", title_fontsize=15,
            handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
# plt.savefig(dir_name + "/fusions_plot" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
# plt.savefig(dir_name + "/fusions_1_high" + ".svg", dpi=1000, format="svg", bbox_inches = 'tight')
plt.show()


mer_ghz = out_df[["mer_ghz_list"]].to_numpy().flatten()
ghz_list = out_df[["ghz_list"]].to_numpy().flatten()
print((mer_ghz/ghz_list))
