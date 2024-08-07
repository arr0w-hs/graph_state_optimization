import sys
import os
from pathlib import Path
#dir_name = os.path.dirname(__file__)
#os.chdir(dir_name)
sys.path.append('..')

import cvxpy as cvx
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
# from ILP_VMinor import has_VM
import time
warnings.simplefilter(action='ignore', category=FutureWarning)  # this is called to suppress an annoying warning from networkx when running a version < 3.0

#

# def linearize(e1, e2):
#     # print(type(e1), type(e2))
#     if type(e1) is int and type(e2) is int:
#         print("!!!!!!!!!!!")
#         return e1*e2, []
#     elif type(e1) is int:
#         if e1 == 1:
#             return e2, []
#         else:
#             return 0, []
#     elif type(e2) is int:
#         print("!!")
#         if e2 == 1:
#             return e1, []
#         else:
#             return 0, []
#     else:
#         e = cvx.Variable(1, boolean=True)
#         constraint1 = (e <= e1)
#         constraint2 = (e <= e2)
#         constraint3 = (e >= e1 + e2 - 1)
#         constraint4 = (e >= 0)
#     return e, [constraint1, constraint2, constraint3, constraint4]


# def linearize(e1, e2):
#     # if False:
#     #     pass
#     #print("jj")
#     if type(e1) is int or type(e2) is int:
#         if type(e1) is int and e1 == 0:
#             return 0, []
#         elif type(e2) is int and e2 == 0:
#             return 0, []
#         elif type(e1) is int and e1 == 1:
#             return e2, []
#         elif type(e2) is int and e2 == 1:
#             return e1, []
#     else:
#         e = cvx.Variable(1, boolean=True)
#         constraint1 = (e <= e1)
#         constraint2 = (e <= e2)
#         constraint3 = (e >= e1 + e2 - 1)
#         constraint4 = (e >= 0)
#         return e, [constraint1, constraint2, constraint3, constraint4]

def linearize(e1, e2):
    # if False:
    #     pass
    #print("jj")
    if type(e1) is int or type(e2) is int:
        return e2*e1, []

    #elif type(e2) is int:
    #    return e2*e1, []

    else:
        e = cvx.Variable(1, boolean=True)
        constraint1 = (e <= e1)
        constraint2 = (e <= e2)
        constraint3 = (e >= e1 + e2 - 1)
        constraint4 = (e >= 0)
        return e, [constraint1, constraint2, constraint3, constraint4]


def create_thetap(n, W):
    # this function is used to cast the 1D list
    # of selection variables into a more natural 2D matrix form
    # Purely done for convenience/readability

    # function also return the variable corresponding to the num of edges

    # W is the graph whose edges correspond to the weights

    matrix = dict()
    num_edges = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                sel_variable = cvx.Variable(1, boolean=True, name="e_" + str(i) + '-'+ str(j))
                matrix[i, j] = sel_variable
                matrix[j, i] = sel_variable

                num_edges += W[i, j]*sel_variable
                assert W[i, j] == W[j, i]
            elif i == j:
                matrix[i, j] = 0
    return matrix, num_edges


def reconstruct_thetap(thetap, n):
    # Reconstructs adjacency matrix of optimal values of thetap and its graph
    adj_matrix = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            if i < j:
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


def minimize_edges(input_G, W=None, draw=False, export_only=False, filename=None):
    if export_only and filename is None:
        raise RuntimeError("Export only flag used, but no filename provided.")
    if filename is not None and not export_only:
        RuntimeWarning("Filename provided, but export only flag not used.")
    if export_only and draw:
        RuntimeWarning("Draw flag was set, but problem was not optimized since export_only was set as well.")

    n = len(input_G.nodes())
    # W is the matrix of weights used for each edge
    if W is None:
        W = np.ones((n, n))

    if draw:
        print("Plotting input graph")
        positions = nx.spring_layout(input_G)
        nx.draw(input_G, pos=positions)
        plt.show()

    theta = nx.adjacency_matrix(input_G)
    theta = np.asarray(theta.todense())

    # add check that adj matrix is square

    thetap, num_edges = create_thetap(n, W)

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


    problem = cvx.Problem(cvx.Minimize(num_edges), [*constraints_type1, *constraints_type2,
                                                        *constraints_type3, *constraints_type4])
    if not export_only:
        # attempt to solve
        time1 = time.time()
        problem.solve(solver='MOSEK', mosek_params={'MSK_IPAR_MIO_HEURISTIC_LEVEL': 1,
                                                    'MSK_IPAR_MIO_MAX_NUM_SOLUTIONS': 500})
        time2 = time.time()
        # print("Time diff = ", time2-time1)
        # problem.solve()
        if problem.status != "optimal":
            print(problem.status)

        adj_matrix, G = reconstruct_thetap(thetap, n)

        if draw:
            print("Plotting output graph")
            nx.draw(G, pos=positions)
            plt.show()
        return G, problem.value
    else:
        # import pdb
        # pdb.set_trace()
        # problem.writedata(filename + ".task")
        # x = problem.get_problem_data(solver='MOSEK')
        # print(x)

        return None, "Problem exported."



#
if __name__ == "__main__":
    from graphs import get_graph_dict, LC
    # n = 8
    # p = 0.2
    #
    # G = nx.sedgewick_maze_graph()
    # verts = [0, 1, 2, 3, 4, 5]
    #
    # for v in verts:
    #     G = LC(G, v)

    # G = nx.cycle_graph(8)
    # verts = [0, 1, 2, 3, 4, 5, 6]
    # for v in verts:
    #     G = LC(G, v)
    G2 = nx.Graph()
    G2.add_edges_from(
        [(0, 2), (0, 6), (0, 8), (0, 10), (0, 14), (0, 16), (1, 3), (1, 4), (1, 6), (1, 9), (1, 10), (1, 13), (1, 15),
         (1, 16), (2, 4), (2, 5), (2, 7), (2,
                                           10), (2, 11), (2, 14), (2, 16), (3, 8), (3, 9), (3, 10), (3, 11), (3, 16),
         (4, 6), (4, 7), (4, 9), (4, 12), (4, 13), (4, 16), (5, 6), (5, 7), (5, 8),
         (5, 9), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 8), (6, 10), (6, 11), (6, 12), (6, 14), (6, 16),
         (7, 9), (7, 10), (7, 12), (7, 15), (7,
                                             16), (8, 10), (8, 11), (8, 13), (8, 16), (9, 10), (9, 11), (9, 14),
         (9, 15), (9, 16), (10, 13), (10, 16), (11, 13), (11, 14), (11, 16), (12, 14),
         (12, 16), (13, 15), (13, 16), (14, 16), (15, 16)])

    # G = nx.induced_subgraph(G, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # nx.draw(G)
    # plt.show()
    #
    # for v in verts:
    #     G = LC(G, v)
    # nx.draw(G)
    # plt.show()

    # minimize_edges(G, draw=False, export_only=False,
    #                filename="ER_n_" + str(n) + "_p_" + str(p))
    n = 8
    p = 0.4


    graph_dict = get_graph_dict()
    # G = graph_dict[18]

    # _, num_edges = minimize_edges(G, draw=False, export_only=True, filename="sedgewick")

    times=[]
    N = 1
    # print(len(G2.edges()))
    # print("__")

    edge_dict = {1: [2, 4, 7],
 2: [1, 5, 6],
 3: [4, 7, 11, 13, 14],
 4: [1, 3, 5],
 5: [2, 4, 11, 13, 14],
 6: [2, 7, 11, 13, 14],
 7: [1, 3, 6],
 8: [9, 11, 14],
 9: [10, 12, 8],
 10: [9, 13, 14],
 11: [3, 5, 6, 12, 8],
 12: [9, 11, 13],
 13: [3, 5, 6, 10, 12],
 14: [3, 5, 6, 10, 8]}

    edge_dict = {1: [2, 4, 7],
                 2: [1, 5, 6],
                 3: [4, 13, 14],
                 4: [1, 3, 5],
                 5: [2, 4, 11],
                 6: [2, 7],
                 7: [1, 6],
                 8: [9, 11],
                 9: [12, 8],
                 10: [13, 14],
                 11: [5, 12, 8],
                 12: [9, 11],
                 13: [3, 10],
                 14: [3, 10]}

    G = nx.Graph(edge_dict)
    # G.add_edges_from(edge_dict)
    print(len(G.edges()))
    random.seed(0)
    for n in [13,13, 13, 13, 13]:
        # print(i/N)
        # i = 9
        # G = nx.induced_subgraph(G2, range(i))
        # G = nx.sedgewick_maze_graph()
        # G = nx.complete_graph(3)
        print(len(G.edges()))
        time1 = time.time()

        G = nx.erdos_renyi_graph(n, p)
        # G = nx.complete_graph(10)
        # # print(len(G.edges()))
        # G, num_edges = minimize_edges(G, draw=False)
        #has_VM(G, [0, 1, 2, 3,4], False, True)
        # print(num_edges)
        time2 = time.time()
        times.append(time2-time1)
        # print(num_edges)
        print(times)
        # print(i)
        # print(time2-time1)
        # print("------")
    print(times)
    print(list(G.edges()))


