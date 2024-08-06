import cvxpy as cvx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from random import choice
from graphs import get_graph_dict, LC
from ILP_minimize_edges import reconstruct_thetap, minimize_edges
import time
warnings.simplefilter(action='ignore', category=FutureWarning)  # this is called to suppress an annoying warning from networkx when running a version < 3.0


def linearize_SDP(e1, e2):
    #TODO: update to deal with case where one of e1, e2 is an int and zero
    e = cvx.Variable(1, boolean=True)
    constraint1 = (e <= e1)
    constraint2 = (e <= e2)
    constraint3 = (e >= e1 + e2 - 1)
    constraint4 = (e >= 0)
    return e, [constraint1, constraint2, constraint3, constraint4]


def create_thetap_SDP(n):
    # this function is used to cast the 1D list
    # of selection variables into a more natural 2D matrix form
    # Purely done for convenience/readability

    # function also return the variable corresponding to the num of edges

    matrix = dict()
    num_edges = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                sel_variable = cvx.Variable(1, boolean=True, name="e_" + str(i) + '-'+ str(j))
                matrix[i, j] = sel_variable
                matrix[j, i] = sel_variable

                num_edges += sel_variable
            elif i == j:
                matrix[i, j] = 0
    return matrix, num_edges


def minimize_edges_SDP(input_G, draw=False):
    if draw:
        print("Plotting input graph")
        nx.draw(input_G)
        plt.show()

    theta = nx.adjacency_matrix(input_G)
    theta = np.asarray(theta.todense())

    # add check that adj matrix is square

    n = len(theta)

    thetap, num_edges = create_thetap_SDP(n)

    a = cvx.Variable(n)
    b = cvx.Variable(n)
    c = cvx.Variable(n)
    d = cvx.Variable(n)

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
                    e, e_constraints = linearize_SDP(thetap[i, k], c[i])
                    constraints_type2 += e_constraints
                    constraint += e

            if theta[j][k] == 1: constraint += a[k]

            e, e_constraints = linearize_SDP(thetap[j, k], d[j])
            constraints_type2 += e_constraints
            constraint += e

            if j == k: constraint += b[j]

            constraints_type1.append(constraint == 2*cvx.Variable(1, boolean=True))

    # set constraints of type 3 and 4
    for i in range(n):
        ad_term, ad_constraints = linearize_SDP(a[i], d[i])
        bc_term, bc_constraints = linearize_SDP(b[i], c[i])
        constraints_type3.append(ad_term + bc_term == 1)
        constraints_type4 += ad_constraints
        constraints_type4 += bc_constraints

    # attempt to solve
    problem = cvx.Problem(cvx.Minimize(num_edges), [*constraints_type1, *constraints_type2,
                                                    *constraints_type3, *constraints_type4])

    problem.solve()
    if problem.status != "optimal":
        print(problem.status)

    adj_matrix, G = reconstruct_thetap(thetap, n)

    if draw:
        print("Plotting output graph")
        nx.draw(G)
        plt.show()
    return G, problem.value



if __name__ == "__main__":
    G_dict = get_graph_dict()

    for _ in range(10):
        G = nx.complete_graph(10)
        verts = list(G)
        num_edges_input = len(G.edges())
        time1 = time.time()
        G_output, min_num_edges = minimize_edges_SDP(G, draw=False)
        time2 = time.time()
        print(time2-time1)
        print(len(G_output.edges()))
        nx.draw(G_output)
        plt.show()
        num_edges_output = len(G_output.edges())
        if num_edges_output != len(G.nodes())-1:
            print("Failed :(")
        else:
            pass