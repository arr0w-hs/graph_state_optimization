# Python packages
import os
import csv
import sympy as sp
import numpy as np
import networkx as nx
import itertools as it
from pprint import pprint
from ast import literal_eval
from subprocess import check_output
# Local modules
from .utils import canonical_edge_order, flatten, powerset

bin2gate = {(1, 0, 0, 1): 'I', (0, 1, 1, 0): 'H', (1, 0, 1, 1): 'S',
            (1, 1, 1, 0): 'HS', (0, 1, 1, 1): 'SH', (1, 1, 0, 1): 'HSH'}


def get_adjacency_matrix(graph):
    """ Returns the adjacency matrix with a canonical node basis """
    # Canonically orders the nodes and edges
    key = sorted(graph.nodes())
    edges = canonical_edge_order(graph.edges())
    # Creates the adjacency matrix and exports to CSV
    adj_mat = np.array([[int(tuple(sorted((u, v))) in edges) for u in key]
                        for v in key])
    return adj_mat, key


def export_adjacency_matrix(graph, filename):
    """ Exports adjacency matrix to CSV file """
    # Gets adjacency matrix
    adj_mat, key = get_adjacency_matrix(graph)
    # Writes to CSV file
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(adj_mat)
    return adj_mat, key


def to_rref(A):
    """
        Takes n x m matrix A to it's reduced row echelon form.
        Algorithm from: https://www.di-mgt.com.au/matrixtransform.html
    """
    n, m = A.shape
    j = 0
    for i in range(n):
        # While column j has all zero elements, set j = j+1. If j>m return A.
        # print i, j
        while all(A[i:, j] == 0):
            j += 1
            if j >= m:
                return A
        # If element a_ij = 0, then swap row i with row x>i where a_xj != 0.
        if A[i, j] == 0:
            for x in range(i+1, n):
                if A[x, j] != 0:
                    # print x
                    A[[x, i]] = A[[i, x]]
                    break
        # Divide each element of row i by a_ij, thus making the pivot a_ij = 1.
        A[i] = A[i]/A[i, j]
        # For each row k from 1 to n, with k != i,
        # subtract row i multiplied by a_kj from row k.
        for k in [k for k in range(n) if k != i]:
            A[k] = (A[k] - A[i] * A[k, j]) % 2
    return A


def GF2nullspace(A):
    """
        Finds nullspace of A using RREF(A).
        Follows decription here:
        https://math.stackexchange.com/questions/130207/
        finding-null-space-basis-over-a-finite-field
    """
    # Takes A to reduced row echelon form and removes any all-zero rows
    A = to_rref(A)
    A = A[~(A == 0).all(1)]
    # Permutes columns of A into [ I_n | P ] form (I_n is n x n, P is n x k)
    n, m = A.shape
    perms = []
    I = np.eye(n, dtype=int)
    for i in range(n):
        while A[:, i].tolist() != I[:, i].tolist():
            perm = range(i, m)
            A[:, perm] = A[:, list(range(i + 1, m)) + [i]]
            perms.append(perm)
    P = A[:, n:]
    # N(A) is spanned by [P^T | I_k ] (P^T is k x n and I_k is k x k)
    N = np.hstack([P.T, np.eye(P.shape[1], dtype=int)])
    # Undo column permutations to retrieve N(A) in original basis
    for perm in perms[::-1]:
        perm = list(perm)
        N[:, perm] = N[:, [perm[-1]] + perm[:-1]]
    return N


def are_lc_equiv(g1, g2):
    """
        Tests whether two graphs are equivalent up to local complementation.
        If True, also returns every unitary such that |g2> = U|g1>.
    """
    # Gets adjacency matrices and returns false if differing bases
    am1, k1 = get_adjacency_matrix(g1)
    am2, k2 = get_adjacency_matrix(g2)
    dim1, dim2 = len(k1), len(k2)
    if k1 != k2:
        raise RuntimeError("k1 != k2, since " + str(k1) + "!= " + str(k2))
    if am1.shape != (dim1, dim1):
        raise RuntimeError("am1.shape != dim1, since " + str(am1.shape) + "!= " + str(dim1))
    if am2.shape != (dim2, dim2):
        raise RuntimeError("am2.shape != dim2, since " + str(am2.shape) + "!= " + str(dim2))
    # Defines binary matrices
    I = sp.eye(dim1)
    S1 = sp.Matrix(am1).col_join(I)
    S2 = sp.Matrix(am2).col_join(I)
    # Defines symbolic variables
    A = sp.symbols('a:' + str(dim1), bool=True)
    B = sp.symbols('b:' + str(dim1), bool=True)
    C = sp.symbols('c:' + str(dim1), bool=True)
    D = sp.symbols('d:' + str(dim1), bool=True)
    # Defines solution matrix basis
    abcd = flatten(zip(A, B, C, D))
    no_vars = len(abcd)
    no_qubits = no_vars / 4
    # Creates symbolic binary matrix
    A, B, C, D = sp.diag(*A), sp.diag(*B), sp.diag(*C), sp.diag(*D)
    Q = A.row_join(B).col_join(C.row_join(D))
    P = sp.zeros(dim1).row_join(I).col_join(I.row_join(sp.zeros(dim1)))
    # Constructs matrix to solve
    X = [i for i in S1.T * Q.T * P * S2]
    X = np.array([[x.coeff(v) for v in abcd] for x in X], dtype=int)
    # Removes any duplicated and all-zero rows
    X = np.unique(X, axis=0)
    X = X[~(X == 0).all(1)]
    # Finds the solutions (the nullspace of X)
    V = list(GF2nullspace(X))
    if len(V) > 4:
        V = [(v1 + v2) % 2 for v1, v2 in it.combinations(V, 2)]
    else:
        V = [sum(vs) % 2 for vs in powerset(V)]
    assert abs(no_qubits-int(no_qubits)) < 0.001
    no_qubits = int(no_qubits)
    V = [np.reshape(v, (no_qubits, 4)) for v in V]
    V = [v for v in V if all((a * d + b * c) % 2 == 1 for a, b, c, d in v)]
    if V:
        V = [[bin2gate[tuple(r)] for r in v] for v in V]
        return True, V
    else:
        return False, None


if __name__ == '__main__':
    e1 = [(0, 1), (1, 2)]
    g1 = nx.Graph(e1)
    e2 = [(0, 1), (1, 2), (2, 0)]
    g2 = nx.Graph(e2)
    print(are_lc_equiv(g1, g2))

    e1 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3),
          (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    e2 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    g1 = nx.Graph(e1)
    g2 = nx.Graph(e2)
    print(are_lc_equiv(g1, g2))
