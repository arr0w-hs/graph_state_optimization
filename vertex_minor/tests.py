# todo: create better test environment

import sys
import os
from pathlib import Path
#dir_name = os.path.dirname(__file__)
#os.chdir(dir_name)
sys.path.append('..')

from vertex_minor.graphs import get_graph_dict, LC
from optimizer.gsc.is_lc_equiv import are_lc_equiv
from optimizer.ILP_minimize_edges import minimize_edges
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
from vertex_minor.SDP_minimize_edges import minimize_edges_SDP
import pickle


def test_LC_min():
    print("dsdfsg")
    G_dict = get_graph_dict()
    for i, G in enumerate(G_dict.values()):
        verts = list(G)
        num_edges_input = len(G.edges())
        for _ in range(10):
            v = choice(verts)
            G = LC(G, v)

        G_output, min_num_edges = minimize_edges(G, draw=False)
        num_edges_output = len(G_output.edges())
        if num_edges_output != num_edges_input:
            # print(i)
            print("!!!")
            nx.draw(G, with_labels=True)
            plt.show()
            # if i == 1:
            #     with open('counterexample.pickle', 'wb') as handle:
            #         pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)

            nx.draw(G_output, with_labels=True)
            plt.show()
            raise RuntimeError("Did not return minimal number of edges")

def test_are_LC_equivalence():
    G1 = nx.star_graph(3)
    G2 = nx.complete_graph(4)
    G3 = nx.cycle_graph(4)

    assert are_lc_equiv(G1, G2)[0]
    assert are_lc_equiv(G3, G3)[0]
    assert not are_lc_equiv(G2, G3)[0]
    assert not are_lc_equiv(G1, G3)[0]


def test_LC_min_SDP():
    G_dict = get_graph_dict()
    for i, G in enumerate(G_dict.values()):
        verts = list(G)
        num_edges_input = len(G.edges())
        for _ in range(10):
            v = choice(verts)
            G = LC(G, v)

        G_output, min_num_edges = minimize_edges_SDP(G, draw=False)
        num_edges_output = len(G_output.edges())
        if num_edges_output != num_edges_input:
            print(num_edges_output)
            print(num_edges_input)
            # print(i)
            print("!!!")
            nx.draw(G, with_labels=True)
            plt.show()
            # if i == 1:
            #     with open('counterexample.pickle', 'wb') as handle:
            #         pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)

            nx.draw(G_output, with_labels=True)
            plt.show()
            raise RuntimeError("Did not return minimal number of edges")

test_LC_min_SDP()
test_LC_min()
test_are_LC_equivalence()


