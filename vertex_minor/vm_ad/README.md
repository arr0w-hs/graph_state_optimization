# Code for local complementations and vertex-minor

## SAGE

The file GraphClasses.sage contain classes for sage that can be used to study graphs under local complementations and vertex-deletions, i.e. vertex-minors.
Using the class SimpleGraph one can efficiently test whether two graphs are equivalent under local complementations and therefore if their corresponding graph states are equivalent under local Clifford operations.
One can test if a graph is a vertex-minor of another using the algorithms described in https://arxiv.org/abs/1805.05306 and https://arxiv.org/abs/1805.05305. These algorithms can be accessed by using the method-calls `G.has_VM(Gp,method='brute')` or `G.has_VM(Gp,method='DH')`, see below for more details.

Furthermore there are classes for isotropic systems which describe equivalence classes of graphs under local complementations.
Also a class for double occurence words, related to circle graphs and their equivalence classes.
More details of the theory can be found in https://arxiv.org/abs/1805.05306

To use the classes, start sage and load the script by writing `load("GraphClasses.sage")`.

#### Some useful methods:

* `SimpleGraph(data)`: Creates an instance of the class `SimpleGraph`. `data` can be a an instance of the `Graph`-class already in SAGE, a dictionary describing the neighbors of vertices, etc.
* `G1.is_LC_eq(G2)`: Checks if the graph `G1` is LC-equvialent to `G2`, where both graphs are instances of `SimpleGraph`. The method used the algorithm descibed by Bouchet in https://link.springer.com/article/10.1007/BF01275668
* `G.has_VM(Gp,method='brute')`: Checks if `Gp` is a vertex-minor of `G`. Uses the non-efficent algortihm described in https://arxiv.org/abs/1805.05305
* `G.has_VM(Gp,method='DH')`: Checks if `Gp` is a vertex-minor of `G`, where `Gp` is a star graph and `G` is a distance-hereditary graph. Uses the method described in https://arxiv.org/abs/1805.05306
* `G.is_DH()`: Checks if `G` is distance-hereditary. Does this by trying to remove leaves or twins until there is only one vertex left.

#### Example:
```
G=SimpleGraph(range(10))           #Creates a complete graph on the vertices 0,1,...,9
G*0                                #Does a local complementation (\tau) on vertex 0.
G-0                                #Deletes vertex 0
G.meas_seq([0,3],['X','Y'])        #Measures node 0 in X-basis and then node 3 in Y-basis.
G.is_LC_eq(graphs.StarGraph(9))    #Tests if G is LC-equivalent to the star graph.
G.has_VM(graphs.StarGraph(3))      #Tests if G can be measured into a star graph on 4 vertices.
```

#### Graphs used in proofs

In https://arxiv.org/abs/1805.05306 we proved two theorems (Theorem 4.3 and 4.5) by using the fact that a few smaller graphs were not distance-hereditary. The file `Graphs_in_proofs/graphs_in_proof.sage` can be used to check this. Run the file in by typing `sage Graphs_in_proofs/graphs_in_proofs.sage`. This will generate all the graphs used in the proofs of the theorems mentioned and check that these are not distance-hereditary, using the method `is_LC_eq`.

## MONA

The files in MONA implements the monadic second-order formula describing whether a graph is a vertex-minor of another, which we describe in the paper https://arxiv.org/abs/1805.05305
