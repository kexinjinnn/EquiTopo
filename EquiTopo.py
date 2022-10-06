'''
Date: May 2022

This file implements the EquiTopo graphs from:
Communication-Efficient Topologies for Decentralized Learning with $\Om(1)$ Consensus Rate.

'''
import numpy as np
import math
from itertools import chain

def DEquiStatic(n, seed=0, eps=None, p=None, M=None):
    """A function that generates static topology for directed graphs satisfying
        Pr( ||Proj(W)||_2 < eps ) >= 1 - p  
    Args:
        n: number of nodes
        seed: an integer used as the random seed
        eps: the upper bound of l2 norm
        p: the probability that the l2 norm is bigger than eps
        M: conmunnication cost. If M is not given, M is calculated from eps and p. 
    Returns:
        K: a numpy array that specifies the communication topology.
        As: a sequence of basis index
    """
    if M == None:
        M = int(8 * math.log(2 * n / p) / 3 / eps**2)
    # generating M graphs
    np.random.seed(seed)
    As = np.random.choice(np.arange(1, n), size=M, replace=True)
    Ws = np.zeros((n,n))
    for a in As:
        W = np.zeros((n,n))
        for i in range(1, n + 1):
            j =  (i + a) % n
            if j == 0: j = n
            W[i-1, j-1] = (n - 1) / n
            W[i-1, i-1] = 1 / n
        Ws += W
    K = Ws / M
    #assert is_doubly_stochastic(K)
    return K, As

def UEquiStatic(n, seed=0, eps=None, p=None, M=None):
    """A function that generates static topology for undirected graphs satisfying
        Pr( ||Proj(W)||_2 < eps ) >= 1 - p  
    Args:
        n: number of nodes
        seed: an integer used as the random seed
        eps: the upper bound of l2 norm
        p: the probability that the l2 norm is bigger than eps
        M: conmunnication cost. If M is not given, M is calculated from eps and p. 
    Returns:
        K: a numpy array that specifies the communication topology.
        As: a sequence of basis index
    """
    if M == None:
        M = int(8 * math.log(2 * n / p) / 3 / eps**2)
    # generating M graphs
    np.random.seed(seed)
    As = np.random.choice(np.arange(1, n), size=M, replace=True)
    Ws = np.zeros((n,n))
    for a in As:
        W = np.zeros((n,n))
        for i in range(1, n + 1):
            j =  (i + a) % n
            if j == 0: j = n
            W[i-1, j-1] = (n - 1) / n
            W[i-1, i-1] = 1 / n
        Ws += W + W.T
    K = Ws / M / 2
    #assert is_doubly_stochastic(K)
    #assert is_symmetric(K)
    return K, As

def ODEquiDyn(n, Ms, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from D-EquiStatic.
    Args:
        n: number of nodes
        Ms: a sequence of basis index
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(Ms, size=1)
    W = np.zeros((n,n))
    for i in range(1, n + 1):
        j = (i + p) % n
        if j == 0: j = n
        W[i-1, j-1] = (n - 1) / n
        W[i-1, i-1] = 1 / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    return W

def ODEquiDynComplete(n, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from D-EquiStatic with M=n-1.
    Args:
        n: number of nodes
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(np.arange(1, n), size=1)
    W = np.zeros((n,n))
    for i in range(1, n + 1):
        j = (i + p) % n
        if j == 0: j = n
        W[i-1, j-1] = (n - 1) / n
        W[i-1, i-1] = 1 / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    return W

def OUEquiDyn(n, Ms, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from U-EquiStatic.
    Args:
        n: number of nodes
        Ms: a sequence of basis index
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(Ms, size=1)
    s = rng.choice(np.arange(1, n + 1), size=1)
    W = np.zeros((n,n))
    z = np.zeros(n)
    for i in chain(range(int(s), n+1), range(1, int(s))):
        j = (i + p) % n
        if j == 0: j = n
        if z[i-1] == 0 and z[j-1] == 0:
            W[i-1, j-1] = 1
            W[j-1, i-1] = 1
            z[i-1] = 1
            z[j-1] = 1
    for i in range(n):
        if z[i] == 0:
            W[i, i] = 1
    W = np.eye(n) / n + (n - 1) * W / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    #assert is_symmetric(W)
    return W

def OUEquiDynComplete(n, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from U-EquiStatic with M=n-1.
    Args:
        n: number of nodes
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(np.arange(1, n), size=1)
    s = rng.choice(np.arange(1, n + 1), size=1)
    W = np.zeros((n,n))
    z = np.zeros(n)
    for i in chain(range(int(s), n+1), range(1, int(s))):
        j = (i + p) % n
        if j == 0: j = n
        if z[i-1] == 0 and z[j-1] == 0:
            W[i-1, j-1] = 1
            W[j-1, i-1] = 1
            z[i-1] = 1
            z[j-1] = 1
    for i in range(n):
        if z[i] == 0:
            W[i, i] = 1
    W = np.eye(n) / n + (n - 1) * W / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    #assert is_symmetric(W)
    return W


# Unit tests
def equal(a, b, eps=1e-9):
    return abs(a - b) < eps
    
def is_doubly_stochastic(A):
    return (equal(np.sum(A, axis=0), 1.0) | equal(np.sum(A, axis=0), 1.0)).all()

def is_symmetric(A, eps=1e-9):
    return equal(A, A.T, eps=eps).all()