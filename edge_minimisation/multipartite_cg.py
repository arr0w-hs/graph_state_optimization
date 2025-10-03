#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:28:43 2024

@author: hsharma4
"""
import sys
import os
sys.path.append('..')


import matplotlib.pyplot as plt
from math import cos, sin, pi
import networkx as nx
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
import random as random
from gso.edm_sa_ilp import edm_sa_ilp
from gso.edm_sa import EDM_SimAnnealing as sa
from gso.wedm_ilp import minimize_edges as ilp_minimize_edges

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

data_dir = os.path.join(dir_name, "bd_results/data")
plot_dir = os.path.join(dir_name, "bd_results/plots")

def minimize_bundled(ns):

    G = nx.complete_multipartite_graph(*ns)

    print("Initial number of edges: ", len(G.edges()))
    pos = dict()

    groups = len(ns)
    v = 0

    jiggle = True # controls whether to slightly randomly move the vertices around or not
    # colors = ['#D3D3D3', '#C0C0C0','#A9A9A9', '#808080', '#696969']

    colors = ['#ADDBF6', '#ADDBF6','#ADDBF6', '#ADDBF6', '#ADDBF6']
    node_colors = {}
    for group in range(groups):
        color = colors[group]
        for i in range(ns[group]):
            if jiggle:
                perturb_x, perturb_y = 0.6*random.random(), 0.6*random.random()
            else:
                perturb_x, perturb_y = 0, 0
            pos[v] = (5*cos(2*pi*group/groups) + cos(2*pi*i/ns[group]) + perturb_x, 5*sin(2*pi*group/groups)+sin(2*pi*i/ns[group]) + perturb_y)
            node_colors[v] = color
            v += 1

    node_colors2 = []
    for node in G.nodes():
        node_colors2.append(node_colors[node])
    plt.figure()
    nx.draw(G, pos=pos,width = 1, node_color=node_colors2, node_size = 800)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    # plt.show()
    # plt.tight_layout()
    # plt.savefig(plot_dir + "/mcg1" + ".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
    # plt.savefig(plot_dir + "/mcg1" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
    # plt.savefig(plot_dir + "/mcg" + ".png", dpi=800, format="png", bbox_inches = 'tight')


    # sa1 = sa(G, 100, 100)
    # G2, y_list, ui_list = sa1.simulated_annealing("number of edges")
    G2, num_edges, _ = ilp_minimize_edges(G)
    print("Output number of edges: ", len(G2.edges()))
    plt.figure()
    nx.draw(G2, pos=pos,  node_color=node_colors2, node_size = 800)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    # plt.show()
    # plt.tight_layout()
    # plt.savefig(plot_dir + "/mcg_mer" + ".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
    # plt.savefig(plot_dir + "/mcg_mer" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
    # plt.savefig(plot_dir + "/mcg_mer" + ".png", dpi=800, format="png", bbox_inches = 'tight')
    plt.show()

minimize_bundled([4,5,3, 5])

