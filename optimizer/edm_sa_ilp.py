#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:30:13 2024

@author: hsharma4
"""

import sys
import os

dir_name = os.path.dirname(__file__)
os.chdir(dir_name)
sys.path.append('..')

import time


sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from optimizer.edm_sa import EDM_SimAnnealing as SimAnnealing
from optimizer.edm_ilp import ilp_minimize_edges


def edm_sa_ilp(G_in, k_max, temp):

    time1 = time.time()
    sa1 = SimAnnealing(G_in, k_max, temp)
    G_sa, y_list, _ = sa1.simulated_annealing("number of edges")
    sa_edges = (G_sa.number_of_edges())

    # print(len(G.edges()))
    G_final, sa_ilp_edges, ilp_runtime = ilp_minimize_edges(G_sa, draw=False)

    time2 = time.time()

    sa_ilp_runtime = time2-time1
    return G_final, G_sa, sa_edges, sa_ilp_edges, sa_ilp_runtime, ilp_runtime
