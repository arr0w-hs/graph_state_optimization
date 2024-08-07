load("triangular_expanded_graphs.sage")

# Construct the diamond graph
D4 = graphs.DiamondGraph()

# Construct the triangular expansion of the diamond graph
TD4 = triangle_expand(D4)

# Get a Eulerian tour on TD4
eul_tour = TD4.eulerian_tour()

# Construct the circle graph induced by the Eulerian tour
circle_graph = double_occ_word(eul_tour).to_circle_graph()

# Compute the rank-width of this graph
rwd = circle_graph.rank_decomposition()[0]

p = circle_graph.plot(vertex_labels=False, vertex_color='black')
p.save("CETEx_of_diamond.pdf")

print("The rank-width of any circle graph induced by Eulerian tours on the extended trianguler expansion of the diamond graph is: {}".format(rwd))
