from graphstate_opt.edm_sa import EDM_SimAnnealing
import networkx as nx


G = nx.erdos_renyi_graph(5, 0.8)

sa1 = EDM_SimAnnealing(G, 100, 100)
gout, _, _ = sa1.simulated_annealing("number of edges")
print(gout.number_of_edges())
print(G.number_of_edges())