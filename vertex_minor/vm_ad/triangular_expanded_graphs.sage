# The method 'triangle_expand' can be used to get the (extended) triangular expanded graph of a 3-regular/subcubic graph.
# For more details about this concept see https://arxiv.org/abs/1805.05306
#
# Author: Axel Dahlberg

load("GraphClasses.sage")

def four_regularize(G,v):
    neighbors=G.neighbors(v)
    if len(neighbors)==1:
        G.add_edge((v,v))
    elif len(neighbors)==2:
        pass
    elif len(neighbors)==3:
        G.delete_vertex(v)
        (va,vb,vc,vd,ve)=["{}{}".format(v,index) for index in ["a","b","c","d","e"]]
        G.add_vertices([v,va,vb,vc,vd,ve])
        G.add_edges([(v,va),(v,vb),(v,vd),(v,ve),(va,vd),(vb,ve),(vc,vd),(vc,ve),(vd,ve)])
        perm=Permutations([va,vb,vc]).random_element()
        G.add_edges([(perm[i],neighbors[i]) for i in range(3)]*2)
    else:
        raise RuntimeError("len(neighbors)>3")

def triangle_expand(G, only_cubic=False, only_subcubic=True):
    if only_cubic:
        if not all([G.degree(v)==3 for v in G]):
            raise ValueError("all vertices must have degree 3")
    if only_subcubic:
        if not all([G.degree(v)<=3 for v in G]):
            raise ValueError("all vertices must have degree 3")
    H=Graph(G)
    H.allow_multiple_edges(True)
    H.allow_loops(True)
    H.add_edges(H.edges())
    vertices=H.vertices()
    for v in vertices:
        four_regularize(H,v)
    H=four_regular(H)
    H.set_pos(None)
    H.name('')
    return H
