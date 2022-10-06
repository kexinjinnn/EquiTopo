'''
Date: May 2022

This file implements the D/U-EquiStatic topologies from:
Communication-Efficient Topologies for Decentralized Learning with $\Om(1)$ Consensus Rate.

'''
import numpy as np
import math

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


# Unit tests
def equal(a, b, eps=1e-9):
    return abs(a - b) < eps
    
def is_doubly_stochastic(A):
    return (equal(np.sum(A, axis=0), 1.0) | equal(np.sum(A, axis=0), 1.0)).all()

def is_symmetric(A, eps=1e-9):
    return equal(A, A.T, eps=eps).all()