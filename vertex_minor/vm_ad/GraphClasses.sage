load("LC.sage")


class SimpleGraph(Graph):
    def __init__(self,*args,**kwargs):
        """Can be initialized in the same way as Graph(). An additional option is to give a list, which by default will give a complete graph on the vertices provided in the list, if format='empty' is given the graph is instead a graph on the vertices in the list with no edges."""
        try:
            super(SimpleGraph,self).__init__(*args,**kwargs)
            try:
                self._order=list(args[0].get_order())
            except:
                self._order=self.vertices()
        except StandardError as e:
            try:
                data=list(args[0])
                n=len(data)
                if kwargs.has_key('format'):
                    if kwargs['format']=='empty':
                        A=zero_matrix(GF(2),n)
                    else:
                        A=ones_matrix(GF(2),n)+identity_matrix(GF(2),n)
                else:
                    A=ones_matrix(GF(2),n)+identity_matrix(GF(2),n)
                super(SimpleGraph,self).__init__(A)
                self.relabel(data)
                self._order=self.vertices()
            except:
                raise e
    def __add__(self,other):
        """Addition of graphs is given by addition of their corresponding adjacency matrices, or equivalently as the symmetric difference of their edge-sets."""
        Gself=SimpleGraph(self)
        Gother=SimpleGraph(other)
        Vself=Set(Gself.vertices())
        Vother=Set(Gother.vertices())
        Gself.add_vertices(Vother-Vself)
        Gother.add_vertices(Vself-Vother)
        vert_order=Gself.vertices()
        G=SimpleGraph(matrix(GF(2),Gself.adjacency_matrix(vertices=vert_order))+matrix(GF(2),Gother.adjacency_matrix(vertices=vert_order)))
        G.relabel(vert_order)
        return G
    def __sub__(self,other):
        """Subtracting something from a graph means deleting that vertex, if the vertex does not exist an error is raised."""
        G=SimpleGraph(self)
        G.delete_vertex(other)
        G._order.remove(other)
        return G
    def __mul__(self,m):
        """G*v gives the graph where a local complementation is done on vertex v. If v is a list or a string, a sequence of local complementations are performed, starting with the first in the list."""
        G=SimpleGraph(self)
        try:
            for v in m:
                G.tau(v,inplace=True)
        except TypeError:
            G.tau(m,inplace=True)
        return G
    def relabel(self,data):
        super(SimpleGraph,self).relabel(data)
        self._order=self.vertices()
    def get_order(self):
        return self._order
    def set_order(self,order):
        self.relabel(sorted(order))
        self._order=order
    def N(self,X):
        """Returns the symmetric difference of all the neighbor-sets of the vertices in X. If X is a single vertex, the neighbors are returned."""
        try:
            neigh=Set([])
            for v in X:
                neigh^^=self.neighbors(v)
            return neigh
        except TypeError:
            return self.neighbors(X)
    def neighborhood(self,V):
        """Returns union of neighbors of vertices in V"""
        tmp=Set([])
        for v in V:
            tmp+=Set(self.neighbors(v))
        return tmp
    def degree_frequency(self):
        """Returns the degree frequence as a list, [# of vert of dgr=0,.,,,]"""
        dgrs=self.degree()
        n=len(self)
        return map(lambda x:dgrs.count(x),range(n))
    def tau(self,v,inplace=False):
        """Performs a local complementation on vertex v. If inplace, this is done on the graph itself and nothing is returned, otherwise a new graph is returned."""
        if inplace:
            G=self
        else:
            G=SimpleGraph(self)
        try:
            Na=G.neighbors(v)
        except LookupError as err:
            print(v)
            raise err
        Gsub=G.subgraph(Na)
        edges1=Gsub.edges()
        edges2=Gsub.complement().edges()
        G.delete_edges(edges1)
        G.add_edges(edges2)
        G.name('')
        if not inplace:
            return G
    def tau_seq(self,m,inplace=False):
        """Performs a sequence of local complementation of the vertices in m. If inplace, this is done on the graph itself and nothing is returned, otherwise a new graph is returned."""
        if inplace:
            G=self
        else:
            G=SimpleGraph(self)
        for v in m:
            G.tau(v,inplace=True)
        G.name('')
        if not inplace:
            return G
    def meas_Z(self,a,inplace=False):
        """Measures the graph at vertex a in Z-basis. If inplace, this is done on the graph itself and nothing is returned, otherwise a new graph is returned."""
        if inplace:
            G=self
        else:
            G=SimpleGraph(self)
        G.delete_vertex(a)
        G.name('')
        if not inplace:
            return G
    def meas_Y(self,a,inplace=False):
        """Measures the graph at vertex a in Y-basis. If inplace, this is done on the graph itself and nothing is returned, otherwise a new graph is returned."""
        if inplace:
            G=self
        else:
            G=SimpleGraph(self)
        G.tau(a,inplace=True)
        G.delete_vertex(a)
        G.name('')
        if not inplace:
            return G
    def meas_X(self,a,b=None,inplace=False, return_special=False):
        """Measures the graph at vertex a in X-basis. If inplace, this is done on the graph itself and nothing is returned, otherwise a new graph is returned. if b is not specified a neighbor of a is picked. If return_special is True then the choice of the special neighbor of a is returned"""
        if inplace:
            G=self
        else:
            G=SimpleGraph(self)
        if len(G.neighbors(a))==0:
            G.delete_vertex(a)
            bp = None
        else:
            if b==None:
                bp=G.neighbors(a)[0]
            else:
                bp=b
            if not G.has_edge((a,bp)):
                raise ValueError("b in not adjacent to a")
            G.tau_seq([bp,a,bp],inplace=True)
            G.delete_vertex(a)
        G.name('')
        if not inplace:
            if return_special:
                return G, bp
            else:
                return G
        else:
            if return_special:
                return bp
    def meas_seq(self,inds,bases,ret_meas=False,inplace=False):
        """Meas the graph on vertices inds and in bases. For X-measuremnts the first neighbor is choosen as special if there is a neighbor. If inplace, this is done on the graph itself and nothing is returned, otherwise a new graph is returned."""
        if not len(inds)==len(bases):
            raise ValueError("List of vertices and list of bases should be of equal length.")
        if inplace:
            G=self
        else:
            G=SimpleGraph(self)
        if ret_meas:
            meas=""
        for (i,b) in zip(inds,bases):
            if b=="Y":
                G.meas_Y(i,inplace=True)
                if ret_meas:
                    meas+="Y"
            elif b=="Z":
                G.meas_Z(i,inplace=True)
                if ret_meas:
                    meas+="Z"
            elif b=="X":
                if ret_meas:
                    if len(G.neighbors(i))==0:
                        meas+="x"
                    else:
                        meas+=str(G.neighbors(i)[0])
                G.meas_X(i,inplace=True)
            else:
                raise ValueError("Not correct basis")
        if ret_meas:
            return (G,meas)
        return G
    def flip_edge(self,e):
        (u,v)=e
        if (u in self) and (v in self):
            if self.has_edge((u,v)):
                self.delete_edge((u,v))
            else:
                self.add_edge((u,v))
        else:
            raise ValueError(str(e)+" is not a valid edge for this graph")
    def is_LC_eq(self,other,test_Vs=True,allow_disc=False):
        """Tests whether the graphs G1 and G2 are LC-eq, if allow_disc, then disconnected graphs are allowed and the components are tested seperately. If test_Vs=False, the vertex-sets are not tested to be equal."""
        if allow_disc:
            comp_self=Set(map(Set,self.connected_components()))
            comp_other=Set(map(Set,other.connected_components()))
            if comp_self!=comp_other:
                return False
            for s in comp_self:
                if not SimpleGraph(self.subgraph(s)).is_LC_eq(other.subgraph(s)):
                    return False
            return True
        else:
            if not (self.is_connected() and other.is_connected()):
                raise ValueError("Graphs can't be disconnected if allow_disc=False")
            if test_Vs:
                if set(self.vertices())==set(other.vertices()):
                    return test_cond(self.adjacency_matrix(),other.adjacency_matrix())
                else:
                    return False
            return test_cond(self.adjacency_matrix(),other.adjacency_matrix())
    def has_VM(self,other,method='brute',check_DH=False):
        """
        Checks if other is a vertex-minor of self.
        There are currently two method available:
            'brute': A non-efficient algorithm for any graphs, both self and other can be disconnected but can increase the runtime.
                Output: Returns a sequence of vertices 'm' such that the corresponding sequence of local complementations gives a graph which induced subgraph on V(other) is LC-equivalent to other. If no such sequence exists, None is returned.
            'DH': An efficient method for instances where self is distance-hereditary and other is a star graph. If check_DH is set to True then it will be checked whether self is actually distance-hereditary. If other is not a star graph an ValueError is raised.
                Output: Returns a sequence of vertices 'm' such that the corresponding sequence of local complementations gives a graph which induced subgraph on V(other) is a star graph. If no such sequence exists, None is returned.
        """
        
        if method=='brute':
            (ans,bases)=self.can_meas(other)
            if not ans:
                return None

            #Convert meas bases to sequence of local complementations
            m=[]
            for base in bases:
                if base[1]=='Z':
                    pass
                elif base[1]=='Y':
                    m.append(base[0])
                else:
                    m.append(base[0])
                    m.append(int(base[1]))
                    m.append(base[0])
            return m
        elif method=='DH':
            if check_DH:
                if not self.is_DH():
                    raise ValueError("Graph (self) is not distance-hereditary")
            degrees=Set(other.degree())
            n=len(other)
            if not degrees==Set([1,n-1]):
                raise ValueError("Graph (other) is not an star graph")

            try:
                m=self.can_meas_DH(other,check_rwd=False)
                return m
            except RuntimeError:
                return None

    def can_meas(self,other,inds=None,start=True):
        """Brute force test if G can be turned into Gp by measuring vertices in inds in some bases. Returns a (ans,meas) where ans is True of False depending on if it is possible to meas G to Gp or not and meas is a dictionary with which indices should be measured in which basis."""
        try:
            other=graphs.StarGraph(other-1)
        except:
            pass
        self_connected=self.is_connected()
        other_connected=other.is_connected()
        # if not other.is_connected():
        #     raise NotImplementedError("Other graph should be connected")
        if start:
            if inds==None:
                inds=list(Set(self.vertices())-Set(other.vertices()))
            rem_inds=Set(self.vertices())-Set(inds)
            if not rem_inds==Set(other.vertices()):
                raise ValueError("V(Gp)!=V(G)\inds")
        if (not self_connected) and other_connected:
            CC_inds=self.connected_component_containing_vertex(rem_inds.random_element())
            G_sub=self.subgraph(CC_inds)
            try:
                (ans,bases)=G_sub.can_meas(other)
                if not ans:
                    return (False,None)
                else:
                    rest=Set(self.vertices())-Set(CC_inds)
                    for v in rest:
                        bases.append([v,"Z"])
                    return (True,bases)
            except ValueError:
                return (False,None)
        if len(inds)==0:
            if self.is_LC_eq(other):
                return (True,[])
            return (False,None)
        noneq=[]
        for bas in ["Z","Y","X"]:
            (post,meas)=self.meas_seq(inds[:1],[bas],ret_meas=True)
            if post.is_connected() or (not other_connected):
                if not any(post.is_LC_eq(x) for x in noneq):
                    (ans,bases)=post.can_meas(other,inds=inds[1:],start=False)
                    if ans:
                        bases=[[inds[0],meas]]+bases
                        return (True,bases)
                    noneq+=[post]
        return (False,None)
    def can_meas_all(self,other,inds=None):
        """Brute force test if G can be turned into Gp by measuring vertices in inds in some bases, returns all possible sequences"""
        try:
            other=graphs.StarGraph(other-1)
        except:
            pass
        if inds==None:
            inds=list(set(self.vertices())-set(other.vertices()))
        rem_inds=set(self.vertices())-set(inds)
        if not rem_inds==set(other.vertices()):
            print("V(Gp)!=V(G)\inds")
            return (False,None)
        rem_inds=list(rem_inds)
        S=[]
        for bases in Tuples("ZYX",len(inds)):
            (post,meas)=self.meas_seq(inds,bases[::-1],ret_meas=True)
            if post.is_LC_eq(other,allow_disc=True):
                meas_list="".join(meas)
                S+=[meas_list]
        return S
    def connect_leaves(self,V):
        """Returns a sequence of LC such that G[V] has a star graph as subgraph"""
        try: #if V is just the number of vert, makes V=[0,1,2,,,,]
            V=range(V)
        except TypeError:
            V=list(V)
        if len(V)<=1:
            return ([],'star')
        G=SimpleGraph(self)
        t_seq=G.can_meas_DH(V[:-1],check_rwd=False)
        G=(G*t_seq)
        Gsub=G.subgraph(V[:-1])
        for c in Gsub:
            if Gsub.degree(c)==(len(V)-2):
                break
        P=G.shortest_path(V[-1],c)
        if not P:
            raise RuntimeError("Target vertices not connected in graph")
        for v in P[1:-1]:
            leaf_ns=[l for l in V[:-1] if (l!=c and G.has_edge(V[-1],l))]
            if leaf_ns:
                l=leaf_ns.pop()
                G.tau(l,inplace=True)
                t_seq.append(l)
            else:
                G.tau_seq([V[-1],v,V[-1]],inplace=True)
                t_seq+=[V[-1],v,V[-1]]
        dgr_set=Set(G.subgraph(V).degree())
        if dgr_set==Set([1,len(V)-1]):
            conf='star'
        elif dgr_set==Set([len(V)-1]):
            t_seq.append(c)
            conf='star'
        else:
            conf='notstar'
        return (t_seq,conf)
    def is_good(self,c,V,com):
        """Used by can_meas_DH"""
        Gp=self.subgraph([v for v in V if v!=c])
        CC=[v for v in Gp if Gp.degree(v)>0]
        L=[v for v in Gp if Gp.degree(v)==0]
        if self.has_edge((com,c)):
            return False
        for v in L:
            if self.has_edge((com,v)):
                return False
        for v in CC:
            if not self.has_edge((com,v)):
                return False
        return True
    def can_meas_DH(self,V,check_rwd=False):
        """Finds if there is a sequence of LC such that G[V]=S_n and returns sequence if so, otherwise ValueError. G should be distance-hereditary, i.e. rank-width=1"""
        try: #if V is just the number of vert, makes V=[0,1,2,,,,]
            V=range(V)
        except TypeError:
            V=list(V)
        if len(V)<=1:
            return []
        G=SimpleGraph(self)
        if check_rwd:
            if G.rank_decomposition()[0]!=1:
                raise ValueError("rwd has to be 1")
        (t_seq,conf)=G.connect_leaves(V)
        G=(G*t_seq)
        if conf=='star':
            return t_seq
        else:
            k=len(V)
            Gsub=G.subgraph(V)
            for c in Gsub:
                if Gsub.degree(c)==k-1:
                    break
            Vmc=[v for v in V if v!=c]
            Gsubmc=G.subgraph(Vmc)
            u=V[-1]
            CC=[v for v in Vmc if Gsubmc.degree(v)>0]
            L=[v for v in Vmc if not (v in CC)]
            if len(L)==0: #complete directly
                t_seq.append(c)
                if len(CC)==2:
                    return t_seq
                G=(G*[c])
                L.append(u)
                CC.remove(u)
            else: # first star then complete
                common=reduce(lambda x,y:x&y,[Set(G.neighbors(v)) for v in CC],Set(G.vertices()))-Set(V)
                found=False
                for com in common:
                    if not G.has_edge((com,c)):
                        if G.is_good(c,V,com):
                            t_seq+=[com]
                            found=True
                            G=(G*[com])
                    else:
                        common2=(Set(G.neighbors(com))&Set(G.neighbors(c)))-Set(V)
                        for com2 in common2:
                            if not any([G.has_edge((com2,x)) for x in Vmc]):
                                Gtmp=SimpleGraph(G)
                                Gtmp.delete_edge((com,c))
                                if Gtmp.is_good(c,V,com):
                                    t_seq+=[com2,com]
                                    found=True
                                    G=(G*[com2,com])
                                    break
                    if found:
                        break
                if not found:
                    raise RuntimeError("star on V (|V|="+str(len(V))+") is not a vertexminor")
                if len(CC)==2:
                    return t_seq
                L.append(u)
                CC.remove(u)
            #start with complete
            common=reduce(lambda x,y:x&y,[Set(G.neighbors(v)) for v in CC],Set(G.vertices()))-Set(V)
            found=False
            for com in common:
                if not G.has_edge((com,c)):
                    if G.is_good(c,V,com):
                        t_seq+=[com]
                        found=True
                        G=(G*[com])
                else:
                    common2=(Set(G.neighbors(com))&Set(G.neighbors(c)))-Set(V)
                    for com2 in common2:
                        if not any([G.has_edge((com2,x)) for x in Vmc]):
                            Gtmp=SimpleGraph(G)
                            Gtmp.delete_edge((com,c))
                            if Gtmp.is_good(c,V,com):
                                t_seq+=[com2,com]
                                found=True
                                G=(G*[com2,com])
                                break
                if found:
                    break
            if not found:
                raise RuntimeError("star on V (|V|="+str(len(V))+") is not a vertexminor")
            return t_seq
    def to_isotropic(self,Ain=None,Bin=None):
        """Returns the isotropic system of the graph. If the supplementary vectors are not specified they are chosen as A=(w,...,w) and B=(1,...,1)."""
        return Isotropic(self,Ain=Ain,Bin=Bin)
    def eulerian_tour(self):
        """Finds a Eulerian tour of the graph"""
        U_edges=self.eulerian_circuit()
        if not U_edges:
            return False
        return list(map(lambda x:x[0],U_edges))
    def eulerian_tours(self,current=None):
        """
        Find all Eulerian tours.
        """
        outer_call=False
        if current==None:
            current=self.random_vertex()
            outer_call=True
        if self.num_edges()==0:
            return [[]]
        neighbors=self.neighbors(current)
        bridges=[]
        non_bridges=[]
        num_CC=self.connected_components_number()
        for v in neighbors:
            self.delete_edge((current,v))
            if self.connected_components_number()>num_CC:
                bridges.append(v)
            else:
                non_bridges.append(v)
            self.add_edge((current,v))
        tours=[]
        if len(non_bridges)>0:
            for v in non_bridges:
                self.delete_edge((current,v))
                tours+=self.eulerian_tours(v)
                self.add_edge((current,v))
        else:
            for v in bridges:
                self.delete_edge((current,v))
                tours+=self.eulerian_tours(v)
                self.add_edge((current,v))
        map(lambda x:x.insert(0,current),tours)
        if outer_call:
            return list(set(map(double_occ_word,tours)))
        return tours
    def to_double_occ_word(self):
        """
        Computes the double occurence word of the graph. If no such word exists an ValueError is raised.
        (This is a slow brute force algorithm!)
        """
        for P in Permutations(range(len(self))*2):
            m=double_occ_word(P)
            G=m.to_circle_graph()
            if self==G:
                return m
        raise ValueError("This is not a cirle graph")
    def multi_flow(self,sources,targets):
        """Finds maximum number of edge-disjoint paths between sinks and targets"""
        try:
            s=self.add_vertex()
            t=self.add_vertex()
            self.add_edges([(s,si) for si in sources])
            self.add_edges([(t,ti) for ti in targets])
            paths=self.edge_disjoint_paths(s,t)
        except StandardError as err:
            raise err
        finally:
            try:
                self.delete_vertices([s,t])
            except:
                pass
        return map(lambda x:x[1:-1],paths)
    def is_DH(self):
        """Is the graph distance-hereditary, i.e. has rankwidth=1 (efficient)"""
        G=SimpleGraph(self)
        change=True
        while change:
            change=False
            for v in G: #find twin
                if G.degree(v)==1:
                    G.delete_vertex(v)
                    change=True
                    break
            if change:
                continue
            vertices=G.vertices()
            n=len(G)
            for i in range(n):
                for j in range(i+1,n):
                    (v,w)=(vertices[i],vertices[j])
                    if (Set(G.neighbors(v))-Set([w]))==(Set(G.neighbors(w))-Set([v])):
                        G.delete_vertex(v)
                        change=True
                        break
                if change:
                    break
        return G.num_edges()==0
    def neigborhood_vector(self,v):
        """
        Returns a binary vector representing the neighborhood of the vertex v. This is the row of the adjacency matrix, corresponding to the vertex v"
        """
        index=self.vertices().index(v)
        return self.adjacency_matrix()[index]
    def nu(self,x):
        """
        If x is an edge (v,w) this will return a binary vector representing the set (N_v & N_w).
        If x is a set of edges this will be the sum of nu(e) for all e in the set.
        """
        try:
            n=self.num_verts()
            vec=vector(GF(2),[0]*n)
            for e in x:
                vec+=self._nu_internal(e)
            return vec
        except TypeError:
            return self._nu_internal(x)
    def _nu_internal(self,e):
        """
        Used internally by self.nu.
        """
        nx=self.neigborhood_vector(e[0])
        ny=self.neigborhood_vector(e[1])
        return nx.pairwise_product(ny)
    def is_mu(self):
        """
        Property defined Bouchet, used to compute the k-index.
        """
        for v in self: #(i)
            dgr=self.degree(v)
            if dgr%2==0:
                return False
        Gcomp=self.complement()
        for e in Gcomp.edges(): #(ii)
            size_of_nu_e=len(self.nu(e).nonzero_positions())
            if size_of_nu_e%2==1:
                return False
        for C in self.cycles(): #(iii)
            size_of_nu_C=len(self.nu(C).nonzero_positions())
            size_of_C=len(C)
            if (size_of_nu_C+size_of_C)%2==1:
                return False
        return True
    def is_mu_2(self):
        """
        Property defined Bouchet, used to compute the k-index.
        """
        for v in self: #(i)
            dgr=self.degree(v)
            if dgr%2==0:
                return False
        Gcomp=self.complement()
        for e in Gcomp.edges(): #(ii)
            size_of_nu_e=len(self.nu(e).nonzero_positions())
            if size_of_nu_e%2==1:
                return False
        for C in self.cycle_basis(): #(iii)
            size_of_nu_C=len(self.nu(C).nonzero_positions())
            size_of_C=len(C)
            if (size_of_nu_C+size_of_C)%2==1:
                return False
        return True
    def cycle_space(self):
        """Returns a cycle space of the graph"""
        edges=self.edges()
        basis_vecs=[]
        for Cb in self.cycle_basis(output='edge'):
            bas_vec=vector(GF(2),[((e in Cb) or ((e[1],e[0],None) in Cb)) for e in edges])
            basis_vecs.append(bas_vec)
        V=VectorSpace(GF(2),len(edges))
        cycle_space=V.subspace(basis_vecs)
        return cycle_space
    def cycles(self):
        """Returns all cycles of the graph"""
        edges=self.edges()
        cycle_space=self.cycle_space()
        cycles=[]
        for vec in cycle_space:
            C=[edges[i] for i in range(len(edges)) if vec[i]==1]
            cycles.append(C)
        return cycles
    def bineighborhood_space(self):
        """Returns the bineighborhood space as defined by Bouchet"""
        V=VectorSpace(GF(2),self.num_verts())

        #subspace from cycles
        cycle_basis=self.cycle_basis(output='edge')
        basis1=list(map(lambda C:self.nu(C),cycle_basis))
        subspace1=V.subspace(basis1)

        #subspace from edge-sets of complemental graph
        edges_comp=self.complement().edges()
        basis2=list(map(lambda e:self.nu(e),edges_comp))
        subspace2=V.subspace(basis2)

        return subspace1+subspace2
    def k_index(self):
        """Returns the k index of the graph. This is the the number of Eulerian vectors divided by the number of locally equivalent graphs."""
        orthog_dim=self.num_verts()-self.bineighborhood_space().dimension()
        if self.is_mu():
            return 2**orthog_dim+2
        else:
            return 2**orthog_dim
    def number_of_eulerian_vectors(self):
        """Returns the number of Eulerian vectors (slow)"""
        if self.num_verts()==0:
            return 1
        v=self.random_vertex()
        if self.degree(v)==0:
            return 2*(self-v).number_of_eulerian_vectors()
        w=self.neighbors(v)[0]
        G1=self-v
        G2=(self*[v])-v
        G3=(self*[v,w,v])-v
        return sum(map(lambda x:x.number_of_eulerian_vectors(),[G1,G2,G3]))
    def number_of_locally_equivalent_graphs(self):
        """Returns the number of locally equivalent graphs (slow)"""
        k=self.k_index()
        e=self.number_of_eulerian_vectors()
        return e/k

class four_regular(SimpleGraph):
    def __init__(self,*args,**kwargs):
        """
        Inherits the SimpleGraph-class but all vertices must have degree 4
        """
        super(four_regular,self).__init__(*args,**kwargs)
        if not all(map(lambda v:self.degree(v)==4,self.vertices())):
            raise ValueError("All vertices has to have degree 4")
    def merge_transition(self,v,e1,e2):
        """Merge a transition"""
        if not ((v in e1) and (v in e2)):
            raise ValueError("both edges of transition must be incident on v")
        e1=e1[:2]
        e2=e2[:2]
        es=list(map(lambda e:e[:2],[e for e in self.edges() if (v in e)]))
        if len(es)==3:
            neighbors=Set(self.neighbors(v))
            new_e=list(neighbors-Set([v]))
            self.delete_vertex(v)
            if len(new_e)==1:
                self.add_edge((new_e[0],new_e[0]))
            elif len(new_e)==2:
                self.add_edge(new_e)
            else:
                raise RuntimeError("new e is > 2")
        elif len(es)==4:
            try:
                es.remove(e1)
            except ValueError:
                es.remove(e1[::-1])
            try:
                es.remove(e2)
            except ValueError:
                es.remove(e2[::-1])
            (e3,e4)=es
            self.delete_vertex(v)
            new_e1=list(Set(e1+e2)-Set([v]))
            new_e2=list(Set(e3+e4)-Set([v]))
            for e in [new_e1,new_e2]:
                if len(e)==1:
                    self.add_edge((e[0],e[0]))
                elif len(e)==2:
                    self.add_edge(e)
                else:
                    raise RuntimeError("new e is > 2")
        else:
            raise RuntimeError("number of incident edges > 4")

class double_occ_word(object):
    def __init__(self,m):
        """
        A double occurence word. All elements have to occur exactly twice.
        """
        try:
            self._m=list(m)
            self._V=set(self._m)
            self._n=len(self._V)
            if not all([self._m.count(v)==2 for v in self._V]):
                raise ValueError("All elements have to occur exactly twice")
        except TypeError:
            self._n=int(m)
            S=SymmetricGroup(2*self._n)
            perm=Permutation(S.random_element())
            self._m=perm.action(range(self._n)*2)
            self._V=set(self._m)
    def __repr__(self):
        return str(self._m)
    def __len__(self):
        return self._n
    def __iter__(self):
        return iter(self._m)
    def __getitem__(self,i):
        return self._m[i]
    def __mul__(self,other):
        """Performs a k-transformation at other or the elements of other (Kotzig transformation)"""
        try:
            tau_seq=list(other)
            m=double_occ_word(self)
            for v in tau_seq:
                m=m.tau(v)
            return m
        except TypeError:
            return self.tau(other)
    def __sub__(self,other):
        """Removes the instances of other or the elements of other"""
        try:
            to_del=list(other)
            m=double_occ_word([v for v in self if not (v in to_del)])
            return m
        except TypeError:
            m=double_occ_word([v for v in self if not (v==other)])
            return m
    def __eq__(self,other):
        """Tests for equivalence up to cyclic permutations and reversal"""
        if not type(self)==type(other):
            raise TypeError("Both objects have to be double occurence words")
        v=self[0]
        (s,e)=other.indices(v)
        for i in [s,e]:
            if self._m==(other[i:]+other[:i]):
                return True
            if self._m==(other[i::-1]+other[:i:-1]):
                return True
        return False
    def __ne__(self,other):
        return not self==other
    def __hash__(self):
        return 0
    def tau(self,v):
        """Performs a k-transformation at v (Kotzig transformation)"""
        (s,e)=self.indices(v)
        return double_occ_word(self[:s+1]+self[e-1:s:-1]+self[e:])
    def delete_vertex(self,v):
        (s,e)=self.indices(v)
        self._m.pop(s)
        self._m.pop(e)
        self._V.remove(v)
        self._n-=1
    def delete_vertices(self,verts):
        for v in verts:
            self.delete_vertex(v)
    def indices(self,v):
        s=self._m.index(v)
        e=s+1+self[s+1:].index(v)
        return (s,e)
    def is_alternance(self,u,v):
        """Check if uv is an alternance of self (...u...v...u...v...)"""
        (su,eu)=self.indices(u)
        (sv,ev)=self.indices(v)
        if su<sv:
            if sv<eu:
                if eu<ev:
                    return True
                else:
                    return False
            else:
                return False
        else:
            if su<ev:
                if ev<eu:
                    return True
                else:
                    return False
            else:
                return False
    def alternances(self):
        """Finds the alternances of self"""
        alt=Set()
        for (u,v) in Subsets(self._V,2):
            if self.is_alternance(u,v):
                alt+=Set([(u,v)])
        return alt
    def to_circle_graph(self):
        """Finds the circle graph of the double occurence word"""
        alt=self.alternances()
        G=SimpleGraph(range(self._n),format='empty')
        G.relabel(list(self._V))
        G.add_edges(alt)
        return G
    def to_4regular_graph(self):
        """Finds a 4-regular graph for which self is an Eulerian tour"""
        if self._n>1:
            G=SimpleGraph(graphs.CycleGraph(2*self._n))
            G.allow_multiple_edges(True)
            G.allow_loops(True)
            G.name('')
            perm={} #for relabeling
            for v in self._V:
                ind=self.indices(v)
                loops=0
                if G.has_edge(ind):
                    loops+=1
                    G.delete_edge(ind)
                    if G.has_edge(ind):
                        loops+=1
                    G.add_edge(ind)
                G.merge_vertices(ind)
                G.add_edges([(ind[0],ind[0]) for _ in range(loops)])
                perm[ind[0]]=v
            G.relabel(perm)
            G.set_pos(None)
            return four_regular(G)
        else:
            v=self._m[0]
            G=SimpleGraph([(v,v),(v,v)],format='list_of_edges',loops=True,multiedges=True)
            return four_regular(G)

class K_object(object):
    _f=GF(4,'w')
    _w=_f.gen()
    def __init__(self,a):
        """An element of GF(4)"""
        try:
            self._a=(GF(2)(a[0]),GF(2)(a[1]))
        except:
            try:
                self._a=(GF(2)(a._a[0]),GF(2)(a._a[1]))
            except:
                if a=='I':
                    self._a=(GF(2)(0),GF(2)(0))
                elif a=='X':
                    self._a=(GF(2)(1),GF(2)(0))
                elif a=='Z':
                    self._a=(GF(2)(0),GF(2)(1))
                elif a=='Y':
                    self._a=(GF(2)(1),GF(2)(1))
                else:
                    raise TypeError("a should be consist of two elements of GF(2), a GF(4) element or be 'I','X','Z' or 'Y'")
    def __repr__(self):
        if self._a[0]==0 and self._a[1]==0:
            return 'I'
        elif self._a[0]==1 and self._a[1]==0:
            return 'X'
        elif self._a[0]==0 and self._a[1]==1:
            return 'Z'
        else:
            return 'Y'
    def __add__(self,other):
        return K_object(self.to_vector()+other.to_vector())
    def __eq__(self,other):
        try:
            return self.to_vector()==other.to_vector()
        except:
            return self.to_vector()==other
    def __ne__(self,other):
        return not self==other
    def __hash__(self):
        return hash(self._a)
    def __pow__(self,other):
        """Returns symplectic product (self^2*other+self*other^2)"""
        if self==0 or other==0 or self==other:
            return GF(2)(0)
        else:
            return GF(2)(1)
    def __mul__(self,other):
        try:
            a=self.to_gf4()*other
            return K_object(((K_object._w^2*a).trace(),a.trace()))
        except:
            a=self.to_gf4()*other.to_gf4()
            return K_object(((K_object._w^2*a).trace(),a.trace()))
    def __rmul__(self,other):
        return self*other
    def __iter__(self):
        return iter(self._a)
    def to_vector(self):
        return vector(GF(2),self._a)
    def to_gf4(self):
        return K_object._f(self._a[0]*1+self._a[1]*K_object._w)

class K_vec(object):
    def __init__(self,vec):
        """An element of GF(4)^n"""
        try:
            self._vec=tuple(K_object(a) for a in vec)
        except:
            try:
                self._vec=tuple(K_object((vec[2*i],vec[2*i+1])) for i in range(len(vec)/2))
            except:
                raise TypeError("vec should be a vector over GF(2) of even length")
    def __add__(self,other):
        return K_vec(self.to_vector()+other.to_vector())
    def __pow__(self,other):
        """Symplectic inner product"""
        return sum([self[i]^other[i] for i in range(len(self))],GF(2)(0))
    def __repr__(self):
        return '('+''.join(map(str,self))+')'
    def __len__(self):
        return len(self._vec)
    def __eq__(self,other):
        try:
            return self.to_vector()==other.to_vector()
        except:
            return self.to_vector()==other
    def __ne__(self,other):
        return not self==other
    def __getitem__(self,P):
        if type(P)==type(Set([])):
            vec=()
            for i in range(len(self)):
                if i in P:
                    vec+=(self._vec[i],)
                else:
                    vec+=(K_object((0,0)),)
            return K_vec(vec)
        else:
            try:
                return K_any([self._vec[i] for i in P])
            except:
                return K_any(self._vec[P])
                print("falied")

    def __iter__(self):
        return iter(self._vec)
    def to_vector(self):
        return vector(GF(2),sum([tuple(a) for a in self],()))
    def is_complete(self):
        """Are all elements non-zero"""
        return all([a!=0 for a in self])
    def is_supplementary(self,other):
        """Is the pairwise symplectic product (1,....,1)"""
        return all([self[i]^other[i]==1 for i in range(len(self))])
    def vec_hat(self):
        V=VectorSpace(GF(2),2*len(self))
        return V.subspace_with_basis([self[Set([i])].to_vector() for i in range(len(self))])

class K_mat(object):
    def __init__(self,mat):
        """An element of GF(4)^(nxn)"""
        self._mat=[K_vec(vec) for vec in mat]
    def __repr__(self):
        to_ret=""
        for row in self._mat:
            to_ret+="["+str(row)[1:-1]+"]\n"
        return to_ret[:-1]
    def __add__(self,other):
        return K_mat(self.to_matrix()+other.to_matrix())
    def __mul__(self,other):
        return self.to_matrix()*other
    def __rmul__(self,other):
        return other*self.to_matrix()
    def __iter__(self):
        return iter(self._mat)
    def __len__(self):
        return len(self._mat)
    def __eq__(self,other):
        try:
            return self.to_matrix()==other.to_matrix()
        except:
            return self.to_matrix()==other
    def __ne__(self,other):
        return not self==other
    def __getitem__(self,i):
        try:
            return K_any(self._mat[i])
        except:
            try:
                return K_any([vec[i[1]] for vec in self._mat[i[0]]])
            except:
                return K_any(self._mat[i[0]][i[1]])
    def __setitem__(self,i,row):
        self._mat[i]=row
    def to_matrix(self):
        return matrix(GF(2),[vec.to_vector() for vec in self])
    def transpose(self):
        return K_mat([[self[i,j] for i in range(len(self))] for j in range(len(self[0]))])

def K_any(obj):
    """Returns obj as either a K_mat,K_vec or K_object"""
    try:
        return K_mat(obj)
    except:
        try:
            return K_vec(obj)
        except:
            try:
                return K_object(obj)
            except:
                raise TypeError("Couldn't convert obj to K_mat, K_vec or K_object")

class K(object):
    def __init__(self,n=None,m=None):
        """Space, VectorSpace or Matrixspace over GF(4)"""
        if n==None and m!=None:
            raise ValueError("For there to be columns there has to be rows")
        if n!=None:
            self._n=int(n)
        else:
            self._n=n
        if m!=None:
            self._m=int(m)
        else:
            self._m=m
    def __repr__(self):
        if self._n==None and self._m==None:
            return "Space with elements {I,X,Z,Y}"
        elif self._m==None:
            return "Vectorspace with elements {I,X,Z,Y}^"+str(self._n)
        else:
            return "Matrixspace with elements {I,X,Z,Y}^"+str(self._n)+"x"+str(self._m)
    def __iter__(self):
        if self._n==None and self._m==None:
            return (K_object(x) for x in VectorSpace(GF(2),2))
        elif self._m==None:
            return (K_vec(x) for x in Tuples(list(K()),self._n))
        else:
            return (K_mat(x) for x in Tuples(list(K(self._n)),self._m))

class euler_vec(K_vec):
    def __init__(self,vec,S,check_prop=True):
        """Eulerian vector of the isotropic system S"""
        super(euler_vec,self).__init__(vec)
        self._S=S
        self._vec=list(self._vec)
        if check_prop:
            if not self._S.is_eulerian(self):
                raise ValueError("vec is not a Eulerian vector of S")
    def __getitem__(self,P):
        try:
            vec=()
            for v in range(len(self)):
                if v in P:
                    vec+=(self[v],)
                else:
                    vec+=(K_object((0,0)),)
            return K_vec(vec)
        except TypeError:
            return self._vec[self._S.V().index(P)]
    def __setitem__(self,v,a):
        self._vec[self._S.V().index(v)]=a
    def __mul__(self,m):
        """The switching of self at m, A*m, or (...((A*m[0])*m[1])...)"""
        new=list(self)
        try:
            for v in m:
                i=S.V().index(int(v))
                current=new[i]
                possible=[x for x in map(K_object,[(1,0),(0,1),(1,1)]) if x!=current]
                new[i]=possible[0]
                if not self._S.is_eulerian(new):
                    new[i]=possible[1]
        except TypeError:
            i=S.V().index(m)
            current=new[i]
            possible=[x for x in map(K_object,[(1,0),(0,1),(1,1)]) if x!=current]
            new[i]=possible[0]
            if not self._S.is_eulerian(new):
                new[i]=possible[1]
        return euler_vec(new,S)
    def __div__(self,other):
        """Finds sequence of switchings taking self to other (efficient)"""
        S=self._S
        A=euler_vec(self,S)
        try:
            B=euler_vec(other,S)
        except ValueError:
            raise ValueError("right hand vector is not Eulerian vector of iso-system of left hand")
        left_m=[]
        right_m=[]
        for v in self.V():
            if A[v]!=B[v]:
                if (A*v)[v]==B[v]:
                    left_m+=[v]
                    A=A*v
                elif A[v]==(B*v)[v]:
                    right_m+=[v]
                    B=B*v
                elif (A*v)[v]==(B*v)[v]:
                    left_m+=[v]
                    right_m+=[v]
                    A=A*v
                    B=B*v
                else:
                    raise RuntimeError("no operations can equality at position "+str(v))
        return "".join(map(str,right_m+left_m[::-1]))
    def S(self):
        return self._S
    def V(self):
        return self._S._V

class Isotropic(object):
    def __init__(self,G,Ain=None,Bin=None):
        """Isotropic system (G,A,B). If A and B are not set then A=(w,...,w) and B=(1,...,1)"""
        try:
            G.iso_class()
            self._G=G._G
            self._A=G._A
            self._B=G._B
            self._n=G._n
            self._V=G._V
            self._L=G._L
            self._eulerian_vectors=G._eulerian_vectors
            self._is_isotropy=G._is_isotropy
        except:
            self._G=SimpleGraph(G)
            self._n=self._G.num_verts()
            self._V=self._G.get_order()
            if Ain==None:
                if Bin==None:
                    self._A=K_vec([K_object((0,1))]*self._n)
                    self._B=K_vec([K_object((1,0))]*self._n)
                else:
                    self._A=K_vec(map(lambda x:K_object(x)*K_object._w,Bin))
                    self._B=K_vec(Bin)
            else:
                if Bin==None:
                    self._A=K_vec(Ain)
                    self._B=K_vec(map(lambda x:K_object(x)*K_object._w,Ain))
                else:
                    if not K_vec(A).is_supplementary(K_vec(B)):
                        raise ValueError("A and B has to be supplementary")
                    self._A=K_vec(Ain)
                    self._B=K_vec(Bin)
            v=self._V[0]
            self._L=[self._A[Set(map(lambda x:self._V.index(x),self._G.N(self._V[i])))]+self._B[Set([i])] for i in range(self._n)]
        self._eulerian_vectors=None
        self._is_isotropy=None
    def __iter__(self):
        return iter(self._L)
    def __repr__(self):
        to_print="Isotropic system on vertices V="+str(self._V)+" and spanned by:\n"
        for vec in self:
            to_print+=str(vec)+"\n"
        return to_print[:-1]
    def __len__(self):
        return self._n
    def __getitem__(self,i):
        return self._L[i]
    def iso_class(self):
        """Used by __init__"""
        return True
    def A(self):
        return self._A
    def B(self):
        return self._B
    def G(self):
        return self._G
    def L(self):
        return self._L
    def V(self):
        return self._V
    def is_isotropy(self):
        """Checks if vectorspace is isotropic"""
        if not self._is_isotropy==None:
            return self._is_isotropy
        else:
            for vec1 in self:
                for vec2 in self:
                    if not vec1^vec2==0:
                        self._is_isotropy=False
                        return False
            self._is_isotropy=True
            return True
    def as_subspace(self):
        V=VectorSpace(GF(2),2*len(self))
        return V.subspace_with_basis([vec.to_vector() for vec in self])
    def rank_of_vec(self,vin):
        """Dimensions of intersection of self and vin-hat"""
        v=K_vec(vin)
        V1=self.as_subspace()
        V2=v.vec_hat()
        return (V1.intersection(V2)).dimension()
    def is_eulerian(self,vin):
        v=K_vec(vin)
        if not v.is_complete():
            return False
        if self.rank_of_vec(v)==0:
            return True
        return False
    def eulerian_vectors(self):
        """Finds all Eulerian vectors of self (slow)"""
        if not self._eulerian_vectors==None:
            return self._eulerian_vectors
        else:
            eul=[]
            for v in GF(2)^(2*len(self)):
                if self.is_eulerian(K_vec(v)):
                    eul+=[v]
            self._eulerian_vectors=tuple(eul)
            return self._eulerian_vectors
    def get_B_and_n(self,Ain):
        """Returns new B in (G,A,B) after performing local complementations taking self to Ain"""
        A=K_vec(self._A)
        m=euler_vec(Ain,self)/A
        n={v:Set(self._G.neighbors(v)) for v in self._V}
        B=K_vec(self._B)
        for v_iter in m:
            v=int(v_iter)
            A_new=A+B[Set([self._V.index(v)])]
            B_new=B+A[Set(map(lambda x:self._V.index(x),n[v]))]
            n_new={v:n[v]}
            n_new.update({w:n[w]^^n[v]^^Set([w]) for w in n[v]})
            n_new.update({w:n[w] for w in Set(self._V)-(n[v]+Set([v]))})
            A=A_new
            B=B_new
            n=n_new
        if A!=K_vec(Ain):
            raise RuntimeError("new A is not equal to Ain")
        return (B,n)
    def fundamental_basis(self,Ain):
        """Fundamental basis wrt Ain"""
        (B,n)=self.get_B_and_n(Ain)
        A=K_vec(Ain)
        b={v:A[Set(map(lambda x:self._V.index(x),n[v]))]+B[Set([self._V.index(v)])] for v in self._V}
        return b
    def graphic_presentation(self,Ain):
        """Graph represented by Eulerian vector Ain"""
        m=euler_vec(Ain,self)/self._A
        return self._G*m
    def to_mat(self):
        return K_mat(self.as_subspace().basis_matrix())
    def kernel_w_cols(self,cols):
        mat=self.to_mat()
        B=mat[:,cols].to_matrix().kernel().basis_matrix()
        return K_mat(B*mat)
