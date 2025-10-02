#from ILP_minimize_edges import minimize_edges
from gso.gsc.is_lc_equiv import are_lc_equiv
import networkx as nx
import matplotlib.pyplot as plt
from gso.edm_sa import EDM_SimAnnealing as SimAnnealing

def minimize_BCGL(n1, n2, leaves, check=False, random=False):
    G = nx.complete_bipartite_graph(n1, n2)
    pos = nx.bipartite_layout(G, list(range(n1)))

    if leaves:
        for v in list(G.nodes()):
            G.add_edge(v, str(v)+"_l")
            if v < n1:
                pos[str(v)+"_l"] = (pos[v][0]-1, pos[v][1])
            else:
                pos[str(v)+"_l"] = (pos[v][0]+1, pos[v][1])
    print("Initial number of edges: ", len(G.edges()))
    # nx.draw(G, pos=pos)
    # plt.show()

    #G2, num_edges = minimize_edges(G)

    sa1 = SimAnnealing(G, 100, 100)
    G2, _, _ = sa1.simulated_annealing("number of edges")

    print("Output number of edges: ", len(G2.edges()))
    G2 = nx.relabel_nodes(G2, {i+n1+n2: str(i) + "_l" for i in range(n1+n2)})
    nx.draw(G2, with_labels=True)
    if check:
        mapping = {str(i) + "_l": i+n1+n2 for i in range(n1+n2)}
        G = nx.relabel_nodes(G, mapping)
        G2 = nx.relabel_nodes(G2, mapping)

        assert are_lc_equiv(G, G2)[0]
        _, V = are_lc_equiv(G, G2)
        print(V)
    plt.show()


if __name__ == "__main__":
    n1 = 6
    n2 = 6
    minimize_BCGL(n1, n2, leaves=False, check=True)
