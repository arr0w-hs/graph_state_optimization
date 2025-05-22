#Code for finding the MER for a given input graph
The folder optimizer has all the files for optimisation  
`edm_sa.py` and `edm_sa_ilp.py` are the two files that can be used to find the MERs. They takes in: `G_in`: a `networkx` graph, `k_max`: maximum iterations and `initial_temp`: the initial temperature.  
-`EDM_SimAnnealing` is the class for implementing simulated annealing approach for finding MERs. This is a heuristic algorithm that can go up to graphs with 100 vertices.
-`edm_sa_ilp(G_in, k_max, temp)` is the function for running SA+ILP algorithm. We use it to find MERs for graphs up to 16 vertices  
-ILP_minimize_edges
