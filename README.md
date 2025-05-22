# Code for finding the MER for a given input graph  
The folder optimizer has all the files for optimisation  
`edm_sa.py` and `edm_sa_ilp.py` are the two files that can be used to find the MERs. They takes in: `G_in`: a `networkx` graph, `k_max`: maximum iterations and `initial_temp`: the initial temperature.  
- `EDM_SimAnnealing` is the class for implementing simulated annealing approach for finding MERs. This is a heuristic algorithm that can go up to graphs with 100 vertices.  
- `edm_sa_ilp(G_in, k_max, temp)` is the function for running the `SA+ILP` algorithm. We use it to find MERs for graphs up to 16 vertices.  
- `edm_ilp.py` is the source code for `SA+ILP` algorithm that implements the ILP part of the `SA+ILP` algorithm  

# Edge-minimisation folder
The edge minimsation folder contains the files that can be used to 
- Generate bounded-degree (BD) and Erdos-Renyi (ER) graphs using `gen_bd.py` and `gen_er.py` files respectively.
- Run edge-miniastion of BD and ER graph using `edm_bd_opt.py' and `edm_bd_opt.py' files respectively.
- `plot_bd.py` and `plot_er.py` files for plotting the results for BD and ER graphs respectively.
- The files that start with `grgs_sampling.py` is for sampling gRGS. `grgs_comparison.py` is to analyse and plot the effects of varying fusion probabilities on the resources required to create a gRGS state.
- `weighted_edm.py` shows a use case of the weighted-edge minimization for distributing graph states in a network. It shows the input graph and the final optimized graph

# The tutorial file
`tutorial-1.ipynb` is the file that shows how the aforementioned functions could be used to find MERs. It plots the input graph, approximate MER from `SA` and the exact MER from `SA+ILP`. It also prints the runtime of the `SA+ILP` and `ILP` algorithms.
