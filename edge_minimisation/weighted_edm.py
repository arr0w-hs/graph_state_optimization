
import sys
import os
from gso.wedm_ilp import minimize_edges
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

plt.rcParams.update({'font.size': 12})
def create_network():
    network = nx.Graph()
    network_labels = nx.Graph()

    city_labels = {'Amsterdam': (11.414518050292742, 22.633961668113237),
             'Delft': (-25.43902943488011, -18.19269675206744),
            'The Hague': (-29.465724145242376, -11.649779448196355),
            'Leiden': (-16.088463947542294, -1.3262728219017257),
            'Utrecht': (27.26176346384192, -8.043362953223312),
            'Rotterdam': (-16.883063986469878, -27.82184969272911)}

    cities = {0: (11.414518050292742, 22.633961668113237),
                   1: (-25.43902943488011, -18.19269675206744),
                   2: (-29.465724145242376, -11.649779448196355),
                   3: (-16.088463947542294, -1.3262728219017257),
                   4: (27.26176346384192, -8.043362953223312),
                   5: (-16.883063986469878, -27.82184969272911)}

    network.add_nodes_from(cities.keys())

    network_labels.add_nodes_from(city_labels.keys())

    # for u, v in itertools.combinations(cities.keys(), 2):
    #
    network.add_edge(2, 3, weight=3)
    network.add_edge(3, 0, weight=7)
    network.add_edge(0, 4, weight=6)
    network.add_edge(5, 4, weight=10)
    network.add_edge(5, 3, weight=5)
    network.add_edge(5, 1, weight=2)
    network.add_edge(2, 1, weight=1)
    network.add_edge(3, 4, weight=9)
    network.add_edge(1, 3, weight=4)
    network.add_edge(0, 5, weight=20)

    plt.figure(1)
    nx.draw(network, pos=cities)
    nx.draw_networkx_labels(network_labels, city_labels)

    # nx.draw_networkx_edges(network, pos=cities)
    edge_labels = nx.get_edge_attributes(network, "weight")
    nx.draw_networkx_edge_labels(network, cities, edge_labels)

    G = nx.Graph()
    G.add_nodes_from(cities.keys())

    G.add_edges_from([(0, 4),
                      (4, 5),
                      (5, 1),
                      (1, 2),
                      (2, 3),
                      (3, 0),
                      (3, 4),
                      (0, 5),
                      (3, 5)])

    W = nx.to_numpy_array(network)
    W[W==0] = 999

    H, num_edges = minimize_edges(G, W=W, draw=False)
    print(num_edges)

    nx.draw_networkx_edges(G, pos=cities, edge_color='r', width=8, alpha=0.5)
    # plt.savefig(dir_name+"/wedm_results/input" + ".pdf", dpi=800, format="pdf", bbox_inches = 'tight')

    plt.figure(2)

    nx.draw(network, pos=cities)
    nx.draw_networkx_labels(network_labels, city_labels)

    # nx.draw_networkx_edges(network, pos=cities)
    edge_labels = nx.get_edge_attributes(network, "weight")
    nx.draw_networkx_edge_labels(network, cities, edge_labels)

    nx.draw_networkx_edges(H, pos=cities, edge_color='g', width=8, alpha=0.5)


    # plt.savefig(dir_name+"/wedm_results/output" + ".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
    # plt.show()

    # print(dir_name)
if __name__ == "__main__":
    create_network()
