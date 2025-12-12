
import os
import graphstate_opt as gso
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings(
    "ignore",
    message="Argument subj in putaijlist64"
)
warnings.filterwarnings(
    "ignore",
    message="Argument subi in putaijlist64: Incorrect array format causing data to be copied"
)
dir_name = os.path.dirname(__file__)


for n in range(16, 17):
    graph = nx.erdos_renyi_graph(n, 0.83)
    print(graph.edges())
    # g_sa, _, _ = gso.edm_sa(graph, 100, 100)

    g_sailp, g_sa, *extra = gso.edm_sa_ilp(graph, 100, 100, solver="moesk")

    print(g_sailp.edges())
    print(g_sa.edges())
    print(g_sailp.number_of_edges())
    print(g_sa.number_of_edges())


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    graphs = [graph, g_sa, g_sailp]

    for ax, G in zip(axes, graphs):
        pos = nx.spring_layout(G)   # or any layout you like
        nx.draw(G, pos, ax=ax, with_labels=False, node_color="#abdbf8", edgecolors = "black", node_size=500)
        ax.set_title(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    plt.tight_layout()
    # plt.savefig(dir_name + "/edge_with_band_"+str(n) + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
    plt.show()
