import matplotlib.image as mpimg
import pprint
import time
from copy import copy
from numpy import array, zeros, float, dot
from random import randint
import math
import numpy as np
import scipy
import sys
from scipy import linalg
import networkx as nx
import matplotlib.pyplot as plt


def gauss_jordan(A, b):
    """
    Returns the vector x such that Ax=b.

    A is assumed to be an n by n matrix and b an n-element vector.
    """
    n, m = A.shape
    # should put in some checks of our assumptions, e.g. n == m
    C = zeros((n, m+1), float)
    C[:, 0:n], C[:, n] = A, b

    for j in range(n):
        # First, do partial pivoting.
        p = j  # the current diagonal element is pivot by default
        # look for alternate pivot by searching for largest element in column
        for i in range(j+1, n):
            if abs(C[i, j]) > abs(C[p, j]):
                p = i
        if abs(C[p, j]) < 1.0e-16:
            print("matrix is (likely) singular")
            return b
        # swap rows to get largest magnitude element on the diagonal
        C[p, :], C[j, :] = copy(C[j, :]), copy(C[p, :])
        # Now, do scaling and elimination.
        pivot = C[j, j]
        C[j, :] = C[j, :] / pivot
        for i in range(n):
            if i == j:
                continue
            C[i, :] = C[i, :] - C[i, j]*C[j, :]
    I, x = C[:, 0:n], C[:, n]
    return x


def print_times(n):

    A = np.random.randint(0, 10, (n, n))
    R = np.random.randint(0, 50, n)
    for i in range(n):
        A[i][i] = randint(1, 3)

    start = time.time()
    M = gauss_jordan(A, R)
    end = time.time()

    start_numpy = time.time()
    M2 = np.linalg.solve(A, R)
    end_numpy = time.time()

    lst_time = time.time()
    M3 = np.linalg.lstsq(A, R, rcond=None)
    lst_end_time = time.time()

    scipy_start = time.time()
    linalg.solve(A, R)
    scipt_end = time.time()

    print(f"My own algorithm: {end-start} ms")
    print(f"Numpy linalg.solve: {end_numpy-start_numpy} ms")
    print(f"Numpy linalg lstsq: {lst_end_time-lst_time} ms")
    print(f"Scipy linalg.solve: {scipt_end-scipy_start} ms")


print_times(100)
# P, L, U = linalg.lu(M)


def LU_partial_decomposition(matrix):
    n, m = matrix.shape
    P = np.identity(n)
    L = np.identity(n)
    U = matrix.copy()
    PF = np.identity(n)
    LF = np.zeros((n, n))
    for k in range(0, n - 1):
        index = np.argmax(abs(U[k:, k]))
        index = index + k
        if index != k:
            P = np.identity(n)
            P[[index, k], k:n] = P[[k, index], k:n]
            U[[index, k], k:n] = U[[k, index], k:n]
            PF = np.dot(P, PF)
            LF = np.dot(P, LF)
        L = np.identity(n)
        for j in range(k+1, n):
            L[j, k] = -(U[j, k] / U[k, k])
            LF[j, k] = (U[j, k] / U[k, k])
        U = np.dot(L, U)
    np.fill_diagonal(LF, 1)
    return PF, LF, U


# n = 10
# A = np.random.randint(0, 10, (n, n))
# R = np.random.randint(0, 50, n)
# for i in range(n):
#     A[i][i] = randint(1, 3)
# P, L, U = LU_partial_decomposition(A)
# print(U)


class Graph:
    def __init__(self):
        self.edges = []
        (self.s, self.t, self.E) = (0, 0, 0)
        self.G = None

    @staticmethod
    def split_line(line):
        [a, b, c] = line.split(" ")
        return (int(a), int(b), int(c))

    def handle_line(self, line):
        self.edges.append(Graph.split_line(line))

    def read_file(self):
        with open("graph.txt") as f:
            for line in f.readlines():
                self.handle_line(line)
        (self.s, self.t, self.E) = self.edges[-1]
        self.edges = self.edges[:-1]

    def draw(self, label):
        pos = nx.kamada_kawai_layout(self.G)
        plt.figure()
        nx.draw(self.G, pos, edge_color='black', width=1, linewidths=1,
                node_size=500, node_color='pink', alpha=0.8,
                labels={node: node for node in self.G.nodes()})
        labels = {}
        for i, (a, b, c) in enumerate(self.edges):
            labels[(a, b)] = label[i]

        nx.draw_networkx_edge_labels(
            self.G, pos, edge_labels=labels, font_color='red', font_size=7)
        plt.axis('off')
        plt.show()

    def calculate_graph(self, draw_mode=False):
        edges = [[a, b] for (a, b, c) in self.edges]
        self.G = nx.DiGraph(directed=True)
        self.G.add_edges_from(edges)
        labels = [c for (a, b, c) in self.edges]
        if draw_mode:
            self.draw(labels)

    def kirchoff(self):
        self.calculate_graph()
        nodes = []
        for (a, b, R) in self.edges:
            nodes.append(a)
            nodes.append(b)
        nodes = set(nodes)
        graph = np.zeros((len(self.edges)+1, len(self.edges)))
        result = np.zeros((len(self.edges)+1, 1))

        O = []
        for i in range(len(nodes)+1):
            new = []
            for k in range(len(nodes)+1):
                new.append(0)
            O.append(new)

        for i, (a, b, R) in enumerate(self.edges):
            graph[a][i] = 1
            graph[b][i] = -1
            O[a][b] = (R, i, 0)
            O[b][a] = (-R, i, 0)
            if(a == self.s and b == self.t):
                O[a][b] = (R, i, self.E)
                O[b][a] = (-R, i, -self.E)

        for i, cycle in enumerate(nx.simple_cycles(self.G)):
            ee = 0.0
            for j, a in enumerate(cycle):
                b = cycle[(j+1) % len(cycle)]
                op = O[a][b][0]
                index = O[a][b][1]
                ee += O[a][b][2]
                graph[len(nodes)+i][index] = op
            result[len(nodes)+i] = ee

        res = np.linalg.lstsq(graph, result)[0]
        self.draw([round(r[0], 5) for r in res])


g = Graph()
g.read_file()
g.kirchoff()
