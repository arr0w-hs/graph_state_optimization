import sys
import os
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import cvxpy as cvx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import time

from base_sa import SimAnnealing as SimAnnealing
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


def linearize(e1, e2):

    e = cvx.Variable(1, boolean=True)
    constraint1 = (e <= e1)
    constraint2 = (e <= e2)
    constraint3 = (e >= e1 + e2 - 1)
    constraint4 = (e >= 0)
    return e, [constraint1, constraint2, constraint3, constraint4]


def create_thetap(n):
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


def minimize_edges(input_G, draw=False):
    if draw:
        print("Plotting input graph")
        positions = nx.spring_layout(input_G)
        nx.draw(input_G, pos=positions)
        plt.show()

    theta = nx.adjacency_matrix(input_G)
    theta = np.asarray(theta.todense())

    # add check that adj matrix is square

    n = len(theta)

    thetap, num_edges = create_thetap(n)

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

    # attempt to solve
    problem = cvx.Problem(cvx.Minimize(num_edges), [*constraints_type1, *constraints_type2,
                                                    *constraints_type3, *constraints_type4])

    problem.solve()
    if problem.status != "optimal":
        print(problem.status)

    adj_matrix, G = reconstruct_thetap(thetap, n)

    if draw:
        print("Plotting output graph")
        nx.draw(G, pos=positions)
        plt.show()
    return G, problem.value


#
if __name__ == "__main__":
    
    x = []
    y = []
    for j in range(19):
        
        n = 10
        p = 0.05*(j+1)
        x.append(p)
    
        times=0
        N = 10
        
        in_edges = []
        sa_edges = []
        ilp_edges = []
        
        for i in range(N):
            
            
            time1 = time.time()
            G = nx.erdos_renyi_graph(n, p)
            #G = nx.complete_graph(10)
            in_edges.append(G.number_of_edges())
            
            sa1 = SimAnnealing(G, 100, 100)
            G, y_list, ui_list = sa1.simulated_annealing("number of edges")
            sa_edges.append(G.number_of_edges())
            
            # print(len(G.edges()))
            _, num_edges = minimize_edges(G, draw=False)
            ilp_edges.append(num_edges)
            time2 = time.time()
            times += time2-time1
            #print(time2-time1)
            #print(" ")
        print(times/N, "avg")
        y.append(times/N)
    plt.figure()
    plt.plot(x,y)
    #plt.plot(x, in_edges)     
    #plt.plot(x, ilp_edges) 
    #plt.plot(x, sa_edges)  
    plt.ylabel('Number of edges')
    plt.xlabel('Iteration')
    plt.legend(["Initial edges", "ILP", "Simulated annealing"])   