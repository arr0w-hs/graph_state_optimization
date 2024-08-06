import networkx as nx
from itertools import combinations

def LC(G, v):
    Nv = G[v]
    for x, y in combinations(Nv, 2):
        if G.has_edge(x, y):
            G.remove_edge(x, y)
        else:
            G.add_edge(x, y)
    return G

def get_graph_dict():
    G2 = nx.complete_graph(2)
    G3 = nx.star_graph(3)
    G4 = nx.path_graph(4)
    G5 = nx.star_graph(5)
    G6 = nx.path_graph(4)
    G6.add_edge(1, 4)

    G7 = nx.path_graph(5)
    G8 = nx.cycle_graph(5)
    G9 = nx.star_graph(6)
    G10 = nx.star_graph(4)
    G10.add_edge(1, 5)

    G11 = nx.path_graph(4)
    G11.add_edges_from([[1, 4], [2, 5]])
    G12 = nx.path_graph(5)
    G12.add_edge(1, 5)
    G13 = nx.path_graph(5)
    G13.add_edge(2, 5)

    G14 = nx.path_graph(6)

    G15 = nx.path_graph(5)
    G15.add_edges_from([[0, 3], [1, 5]])

    G16 = nx.path_graph(5)
    G16.add_edges_from([[2, 5], [1, 3]])

    G17 = nx.cycle_graph(5)
    G17.add_edge(4, 5)

    G18 = nx.cycle_graph(6)

    G_list = [G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13,
              G14, G15, G16, G17, G18]

    return {i+2: G for i, G in enumerate(G_list)}

