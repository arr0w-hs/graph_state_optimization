# File containing some tests for the functions in GraphClasses.sage
# To run these, type 'sage tests.sage'

load("GraphClasses.sage")

def run_tests():
    tests=[test_tau,test_is_LC_eq,test_has_VM,test_is_DH]
    for test in tests:
        try:
            test()
            print("OK")
            print("\n")
        except AssertionError:
            print("FAIL")
            print("\n")

def test_tau():
    print("Testing local complementation (tau)")
    G=SimpleGraph(graphs.StarGraph(6))
    assert(G*0==SimpleGraph(graphs.CompleteGraph(7)))

def test_is_LC_eq():
    print("Testing function to check for LC-equivalence (is_LC_eq)")
    G=SimpleGraph(graphs.CycleGraph(8))
    for _ in range(10):
        m=[]
        for _ in range(10):
            m.append(G.random_vertex())
        assert(G.is_LC_eq(G*m))

    G1=SimpleGraph(graphs.PathGraph(4))
    G2=SimpleGraph(graphs.CompleteGraph(4))
    assert(not G1.is_LC_eq(G2))

def tmp_has_VM():
    Gp=SimpleGraph(graphs.StarGraph(3))

    # Test brute force alg yes-instaces (connected)
    for _ in range(10):
        G=SimpleGraph(Gp)
        for _ in range(6):
            verts=G.vertices()
            v=G.add_vertex()
            num_edges=randint(1,len(G)-1)
            neighbors=Subsets(verts,num_edges).random_element()
            G.add_edges([(v,n) for n in neighbors])
        m=[]
        for _ in range(10):
            m.append(G.random_vertex())
        m2=(G*m).has_VM(Gp,method='brute')
        print(m2)
        print(((G*m)*m2).subgraph([0,1,2,3]).is_LC_eq(Gp))

    # Test brute force alg yes-instaces (disconnected)
    G=SimpleGraph(Gp)
    for _ in range(6):
        verts=G.vertices()
        v=G.add_vertex()
        num_edges=randint(0,len(G)-1)
        neighbors=Subsets(verts,num_edges).random_element()
        G.add_edges([(v,n) for n in neighbors])
    G.add_vertex()
    m=[]
    for _ in range(10):
        m.append(G.random_vertex())
    m2=(G*m).has_VM(Gp,method='brute')
    print(m2)
    print(((G*m)*m2).subgraph([0,1,2,3]).is_LC_eq(Gp))

    # Test brute force alg no-instaces (connected)
    G=SimpleGraph({0:[1,2,3,4],1:[0],2:[0,3,4],3:[0,2,4],4:[0,2,3]})
    m2=G.has_VM(Gp,method='brute')
    print(m2)
    # print(((G*m)*m2).subgraph([0,1,2,3])==Gp)

def test_has_VM():
    print("Testing function to check for vertex-minors (has_VM)")
    Gp=SimpleGraph(graphs.StarGraph(3))

    # Test brute force alg yes-instaces (connected)
    for _ in range(10):
        G=SimpleGraph(Gp)
        for _ in range(6):
            verts=G.vertices()
            v=G.add_vertex()
            num_edges=randint(1,len(G)-1)
            neighbors=Subsets(verts,num_edges).random_element()
            G.add_edges([(v,n) for n in neighbors])
        m=[]
        for _ in range(10):
            m.append(G.random_vertex())
        m2=(G*m).has_VM(Gp,method='brute')
        assert(((G*m)*m2).subgraph([0,1,2,3]).is_LC_eq(Gp))

    # Test brute force alg yes-instaces (disconnected)
    G=SimpleGraph(Gp)
    for _ in range(6):
        verts=G.vertices()
        v=G.add_vertex()
        num_edges=randint(0,len(G)-1)
        neighbors=Subsets(verts,num_edges).random_element()
        G.add_edges([(v,n) for n in neighbors])
    G.add_vertex()
    m=[]
    for _ in range(10):
        m.append(G.random_vertex())
    m2=(G*m).has_VM(Gp,method='brute')
    assert(((G*m)*m2).subgraph([0,1,2,3]).is_LC_eq(Gp))

    # Test brute force alg no-instaces (connected)
    G=SimpleGraph({0:[1,2,3,4],1:[0],2:[0,3,4],3:[0,2,4],4:[0,2,3]})
    assert(not G.has_VM(Gp,method='brute'))
    # Test DH alg no-instaces (connected)
    assert(not G.has_VM(Gp,method='DH'))

    # Test DH alg yes-instaces (connected)
    for _ in range(10):
        G=SimpleGraph(Gp)
        for _ in range(6):
            b=randint(0,1)
            if b: # add leaf
                u=G.random_vertex()
                v=G.add_vertex()
                G.add_edge((u,v))
            else: # do twin split
                u=G.random_vertex()
                neighbors=G.neighbors(u)
                v=G.add_vertex()
                G.add_edges([(v,n) for n in neighbors])
                b=randint(0,1)
                if b: # true or false twin
                    G.add_edge((u,v))
        m=[]
        for _ in range(10):
            m.append(G.random_vertex())
        m2=(G*m).has_VM(Gp,method='DH')
        assert(((G*m)*m2).subgraph([0,1,2,3]).is_LC_eq(Gp))

def test_is_DH():
    print("Testing function to check if a graph is distance-hereditary (is_DH)")
    for _ in range(10):
        G=SimpleGraph(1)
        for _ in range(10):
            b=randint(0,1)
            if b: # add leaf
                u=G.random_vertex()
                v=G.add_vertex()
                G.add_edge((u,v))
            else: # do twin split
                u=G.random_vertex()
                neighbors=G.neighbors(u)
                v=G.add_vertex()
                G.add_edges([(v,n) for n in neighbors])
                b=randint(0,1)
                if b: # true or false twin
                    G.add_edge((u,v))
        assert(G.is_DH())

    G=SimpleGraph(graphs.CycleGraph(5))
    assert(not G.is_DH())


# Run all tests
run_tests()
# tmp_has_VM()
