from ILP_minimize_edges import minimize_edges
from ILP_SVMinor import has_SVM
from ILP_VMinor import has_VM
from SDP_minimize_edges import minimize_edges_SDP
import networkx as nx
import random
from numpy import mean
import matplotlib.pyplot as plt
import time

from graphs import get_graph_dict
#TODO: we should probably separate/rename this for the plots file

def sample_min_edges_ER_graphs(n, p, N, SPD=False):
    random.seed(0)
    all_min_edges = []
    for _ in range(N):
        G = nx.erdos_renyi_graph(n, p)
        if SPD:
            _ , num_edges = minimize_edges_SDP(G, draw=False)
        else:
            _, num_edges = minimize_edges(G, draw=False)
        all_min_edges.append(num_edges)
    return all_min_edges

def sample_has_SVM_ER_graphs(n, p, N, SPD=False):
    random.seed(0)
    for _ in range(N):
        G = nx.erdos_renyi_graph(n, p)
        if SPD:
            raise RuntimeError("SPD of SVM not implemented yet")
            _ , num_edges = minimize_edges_SDP(G, draw=False)
        else:
            _ = has_SVM(G, [0, 1, 2], 1, draw=False, check_LC=False)
    return

def create_plot_avg_min_edges_ER_graphs(n, p, N, SDP=False):
    random.seed(0)
    mean_min_edges = []
    times = []
    for n1 in range(2, n+1):
        print(n1/(n))
        time1 = time.time()
        mean_min_edges.append(mean(sample_min_edges_ER_graphs(n1, p, N, SDP)))
        time2 = time.time()
        times.append((time2-time1)/N)
    plt.plot(range(2, n+1), mean_min_edges)
    plt.xlabel("n")
    plt.ylabel("Mean minimum number of edges")
    plt.title("ER graphs with p = " + str(p))
    plt.show()

    plt.plot(range(2, n+1), times)
    plt.xlabel("n")
    plt.ylabel("Average runtime")
    plt.title("ER graphs with p = " + str(p))
    plt.show()


def create_plot_avg_SVM_time_ER_graphs(n, p, N, SDP=False):
    random.seed(0)
    # data for p = 0.8, 4SVM:
    #[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    #[0.09708757400512695, 0.15946779251098633, 0.2592369556427002, 1.0496899604797363, 3.359155607223511, 3.316668891906738, 1.979524517059326, 2.7262536525726317, 5.163726711273194, 10.143385219573975, 25.220630979537965, 32.33894267082214, 67.4386743068695, 173.19757251739503, 261.28487448692323]


    times = []
    for n1 in range(4, n+1):
        print(n1/(n))
        time1 = time.time()
        _ = sample_has_SVM_ER_graphs(n1, p, N, SDP)
        time2 = time.time()
        times.append((time2-time1)/N)

        print("n = ", n1)
        print("time = ", times[-1])
    plt.plot(range(4, n+1), times)
    print(list(range(4, n+1)))
    print(times)
    plt.xlabel("n")
    plt.ylabel("Average runtime")
    plt.title("ER graphs with p = " + str(p))
    plt.show()


def create_max_min_edges():

    graph_dict = get_graph_dict()
    n_range = range(2, 12)
    max_edges_list = []


    N = 20

    for n in n_range:
        print(n)
        max_edges = 0
        for G in graph_dict.items():
            G = G[1]
            if len(G.nodes()) == n:
                if len(G.edges()) > max_edges :
                    max_edges = len(G.edges())
        if max_edges == 0:
            for _ in range(N):
                connected=False
                while not connected:
                    G = nx.erdos_renyi_graph(n, 1/5)
                    connected = nx.is_connected(G)

                G, _ = minimize_edges(G)
                if len(G.edges()) > max_edges :
                    max_edges = len(G.edges())
        max_edges_list.append(max_edges)
    plt.plot(n_range, max_edges_list)
    plt.show()


def create_data_for_comparison_plot():
    random.seed(0)
    times = []
    p = 0.42
    for n in [13,13, 13, 13, 13]:
        # print(i/N)
        # i = 9
        # G = nx.induced_subgraph(G2, range(i))
        # G = nx.sedgewick_maze_graph()
        # G = nx.complete_graph(3)
        # print(len(G.edges()))
        time1 = time.time()

        G = nx.erdos_renyi_graph(n, p)
        print(hash(frozenset(G.edges())))
        # G = nx.complete_graph(10)
        # # print(len(G.edges()))
        # G, num_edges = minimize_edges(G, draw=False)

        H = nx.complete_graph(4)
        has_VM(G, H, False, False)

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

if __name__ == "__main__":
    # create_data_for_comparison_plot()
    create_plot_avg_SVM_time_ER_graphs(13, 0.42, 5, SDP=False)
    # create_max_min_edges()
    # times = [0.09708757400512695, 0.15946779251098633, 0.2592369556427002, 1.0496899604797363, 3.359155607223511, 3.316668891906738, 1.979524517059326, 2.7262536525726317, 5.163726711273194, 10.143385219573975, 25.220630979537965, 32.33894267082214, 67.4386743068695, 173.19757251739503, 261.28487448692323]
    # n  = 18
    # plt.semilogy(range(4, n+1), times)
    # print(list(range(4, n+1)))
    # print(times)
    #
    # plt.xlabel("n")
    # plt.ylabel("Average runtime")
    # plt.title("ER graphs with p = " + str(0.8))
    # plt.show()
