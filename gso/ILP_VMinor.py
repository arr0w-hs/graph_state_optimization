import sys
import os
from pathlib import Path
#dir_name = os.path.dirname(__file__)
#os.chdir(dir_name)
sys.path.append('..')

import cvxpy as cvx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import time


#from gso.gsc.is_lc_equiv import are_lc_equiv
from gso.ILP_minimize_edges import linearize

warnings.simplefilter(action='ignore', category=FutureWarning)  # this is called to suppress an annoying warning from networkx when running a version < 3.0

def create_VM_color_map(G, H):

    colors = dict()
    for v in G.nodes():
        if v in H:
            colors[v] = 'r'
        else:
            colors[v] = 'b'
    colors = [colors[v] for v in G.nodes()] ## hacky...
    return colors

def create_thetap_VM(n, H):
    matrix = dict()

    for i in range(n):
        for j in range(n):
            if i < j:
                if i in H and j in H:
                    if H.has_edge(i, j):
                        matrix[i, j] = 1
                        matrix[j, i] = 1
                    else:
                        matrix[i, j] = 0
                        matrix[j, i] = 0
                else:
                    sel_variable = cvx.Variable(1, boolean=True, name="e_" + str(i) + '-' + str(j))
                    matrix[i, j] = sel_variable
                    matrix[j, i] = sel_variable
            elif i == j:
                matrix[i, j] = 0
    return matrix


# TODO: merge with functionality in ILP_minimize_edges
def reconstruct_thetap(thetap, n):
    # Reconstructs adjacency matrix of optimal values of thetap and its graph
    adj_matrix = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            if i < j:
                if type(thetap[i, j]) is int:
                    val = thetap[i, j]
                else:
                    val = thetap[i, j].value
                adj_matrix[i][j] = int(val+1/2)  # rounds values to 0 or 1
                adj_matrix[j][i] = int(val+1/2)  # rounds values to 0 or 1
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    all_rows = range(0, adj_matrix.shape[0])
    for k in all_rows:
        G.add_node(k)
    G.add_edges_from(edges)
    return adj_matrix, G


def has_VM(input_G, H, draw=False, check_LC=False):
    if draw:
        print("Plotting input graph")
        positions = nx.spring_layout(input_G)
        colors = create_VM_color_map(input_G, H)
        nx.draw(input_G, pos=positions, node_color=colors)
        plt.show()

    theta = nx.adjacency_matrix(input_G)
    theta = np.asarray(theta.todense())

    # add check that adj matrix is square

    n = len(theta)

    thetap = create_thetap_VM(n, H)

    a = cvx.Variable(n, boolean=True)
    b = cvx.Variable(n, boolean=True)
    c = cvx.Variable(n, boolean=True)
    d = cvx.Variable(n, boolean=True)

    constraints_type1 = []  # constraints from eq (4) from van den nest
    constraints_type2 = []  # constraints from linearizing eq (4) constraints
    constraints_type3 = []  # constraints from eq (5) from van den nest
    constraints_type4 = []  # constraints from linearizing eq (5) constraints

    # set constraints of type 1 and 2
    for j in range(0, n):  # using zero indexing here, contrary to van den nest
        for k in range(0, n):
            constraint = 0
            for i in range(n):
                if theta[i][j] == 1:
                    e, e_constraints = linearize(thetap[i, k], c[i])
                    constraints_type2 += e_constraints
                    constraint += e

            if theta[j][k] == 1: constraint += a[k]

            e, e_constraints = linearize(thetap[j, k], d[j])
            constraints_type2 += e_constraints
            constraint += e

            if j == k: constraint += b[j]

            constraints_type1.append(constraint == 2*cvx.Variable(1, integer=True))

    # set constraints of type 3 and 4
    for i in range(n):
        ad_term, ad_constraints = linearize(a[i], d[i])
        bc_term, bc_constraints = linearize(b[i], c[i])
        constraints_type3.append(ad_term + bc_term == 1)
        constraints_type4 += ad_constraints
        constraints_type4 += bc_constraints

    # I only want to check whether a feasible solution exists (for now). cvx does not seem to do feasibility only checks,
    # that's why I add a dummy optimization variable. Very hacky!
    dummy_optimization = cvx.Variable(1, boolean=True)
    problem = cvx.Problem(cvx.Minimize(dummy_optimization), [*constraints_type1, *constraints_type2,
                                                             *constraints_type3, *constraints_type4])

    problem.solve()
    if problem.status != "optimal":
        print(problem.status)
        return False, None
    else:
        adj_matrix, G = reconstruct_thetap(thetap, n)
        if check_LC:
            assert are_lc_equiv(input_G, G)
            ind_G = nx.induced_subgraph(G, H.nodes())
            assert set(ind_G.nodes()) == set(H.nodes())
            for e in ind_G.edges():
                assert e in H.edges() or e[::-1] in H.edges()
            for e in H.edges():
                assert e in ind_G.edges() or e[::-1] in ind_G.edges()
            print("LC check passed.")

        if draw:
            print("Plotting output graph")
            nx.draw(G, pos=positions, node_color=colors)
            plt.show()

    return True, G



if __name__ == "__main__":
    n = 5
    p = 0.6

    times = 0
    N = 1
    for _ in range(N):
        time1 = time.time()
        G = nx.erdos_renyi_graph(n, p)
        # print(len(G.edges()))
        H = nx.cycle_graph(6)
        feasible, G_output = has_VM(G, H, check_LC=True)
        time2 = time.time()
        times += time2-time1
        print(feasible)
        print(time2-time1)
    print("avg time:")
    print(times/N)
