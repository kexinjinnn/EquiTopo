import numpy as np
from scipy.sparse import diags

# Ring
def generate_cycle_network(n):
    diag_m1 = (1./3) * np.ones(n - 1)
    diag_0  = (1./3) * np.ones(n)
    diag_p1 = (1./3) * np.ones(n - 1)
    A = diags([diag_m1, diag_0, diag_p1], [-1, 0, 1]).toarray()
    A[0,-1] = 1./3
    A[-1,0] = 1./3
    
    return A


# Grid
def generate_grid_network(n):
    
    n_sqrt = int(np.sqrt(n))
    
    neighbors_list = {}
    degrees = {}
    for k in range(n):
        nb_list = [k]
        
        # case 1: for vertex
        if k == 0:
            nb_list.append(k+1)
            nb_list.append(k+n_sqrt)
        elif k == n_sqrt-1:
            nb_list.append(k-1)
            nb_list.append(k+n_sqrt)
        elif k == n-n_sqrt:
            nb_list.append(k+1)
            nb_list.append(k-n_sqrt)
        elif k == n-1:
            nb_list.append(k-1)
            nb_list.append(k-n_sqrt)
        elif k > 0 and k < n_sqrt-1:
            nb_list.append(k-1)
            nb_list.append(k+1)
            nb_list.append(k+n_sqrt)
        elif k > n-n_sqrt and k < n-1:
            nb_list.append(k-1)
            nb_list.append(k+1)
            nb_list.append(k-n_sqrt)
        elif k%n_sqrt == 0:
            nb_list.append(k-n_sqrt)
            nb_list.append(k+1)
            nb_list.append(k+n_sqrt)
        elif k%n_sqrt == n_sqrt-1:
            nb_list.append(k-n_sqrt)
            nb_list.append(k-1)
            nb_list.append(k+n_sqrt)
        else:
            nb_list.append(k-1)
            nb_list.append(k+1)
            nb_list.append(k-n_sqrt)
            nb_list.append(k+n_sqrt)
            
        neighbors_list[k] = nb_list
        degrees[k] = len(nb_list)
            
    A = np.zeros((n, n))
    for k in range(n):
        for l in neighbors_list[k]:
            if l == k:
                continue            
            A[k,l] = 1./max(degrees[k], degrees[l]) # metropolis_rule
        
    for k in range(n):
        A[k,k] = 1. - np.sum(A[:,k])
        
    return A


# The static exponential graph
def ExponentialGraph(n):

    incidenceMat = np.zeros((n,n))
    base_row = np.zeros(n)
    count_ones = 0
    
    for i in range(n):
        if i&(i-1) == 0:
            base_row[i] = 1
            count_ones += 1
            
    for i in range(n):
        incidenceMat[i,:] = np.roll(base_row,i)
        
    W = incidenceMat/count_ones
    Num_neighbor = count_ones
    
    return W, incidenceMat, Num_neighbor


# The onepeer exponential graph
def OnePeer_ExponentialGraph(n):

    subgraph_list = []
    
    for i in range(1, n):    
        incidenceMat = np.zeros((n,n))
        base_row = np.zeros(n)
        base_row[0] = 1
        if i&(i-1) == 0:
            base_row[i] = 1
            for i in range(n):
                incidenceMat[i,:] = np.roll(base_row,i) 
            subgraph_list.append(incidenceMat/2)
    
    return subgraph_list