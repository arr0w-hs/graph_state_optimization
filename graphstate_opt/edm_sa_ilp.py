#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:30:13 2024

@author: hsharma4
"""

import time
import networkx as nx
from typing import Optional
from graphstate_opt.edm_sa import edm_sa
from graphstate_opt.edm_ilp import edm_ilp


def edm_sa_ilp(G_in : nx.Graph, k_max : int, temp: float, solver: Optional[str] = None, draw: bool = False):
    """
    Find the Minimum Edge Representative (MER) for a given graph using SA+ILP.

    This function uses edm_sa as a preprocsssing step before solving the ILP
    to find the MER of an input NetworkX graph. It can use either the
    default CVXPY solver or MOSEK (if licensed and installed).

    Parameters
    ----------
    in_graph : nx.Graph
        Input graph whose MER is to be found.
    k_max : int
        Maximum number of iterations of the simulated annealing (SA) algorithm.
    temp : float
        Initial temperature of simulated annealing.
    solver : str, optional
        Solver to use. Defaults to None, which uses the default CVXPY solver.
        Set to "mosek" to use the MOSEK solver (requires a valid MOSEK license).
    draw : bool, optional
        If True, visualize the input and resulting MER graph. Defaults to False.

    Returns
    -------
    G_final : nx.Graph
        MER graph obtained from the ILP.
    G_sa : nx.Graph
        Approximate MER (~MER) obtained from SA preprocessing step.
    sa_edges : float
        Number of edges in the ~MER graph obtained from SA.
    sa_ilp_edges : float
        Number of edges in MER obtained from SA+ILP
    sa_ilp_runtime : float
        Runtime of the SA+ILP solver, in seconds.
    ilp_runtime : float
        Runtime of the ILP part of SA+ILP, in seconds.

    Notes
    -----
    The function requires CVXPY to be installed. If using MOSEK, ensure that
    a valid MOSEK license is available in the environment.
    """

    """
    function to find the MER for a given graph
    using SA+ILP and ILP. this returns the statistics of
    different algorithms
    inputs: 
        (1) graph to be minimised
        (2) k_max: maximum number of iterations
        (3) temp: initial temperature of the graph

    outputs:
        (1) output graph from SA+ILP
        (2) output graph from SA
        (3) number of edges in graph obtained from SA
        (4) number of edges in graph obtained from SA+ILP
        (5) runtime of SA+ILP
        (6) runtime of ILP

    """
    time1 = time.time()
    G_sa, y_list, _ = edm_sa(G_in, k_max, temp)
    sa_edges = (G_sa.number_of_edges())

    # print(len(G.edges()))
    G_final, sa_ilp_edges, ilp_runtime = edm_ilp(G_sa, solver, draw)

    time2 = time.time()

    sa_ilp_runtime = time2-time1
    return G_final, G_sa, sa_edges, sa_ilp_edges, sa_ilp_runtime, ilp_runtime


if __name__ == "__main__":
    G = nx.erdos_renyi_graph(5, 0.8)
    out = edm_sa_ilp(G, 100, 100)
    print(out)