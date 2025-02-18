import numpy as np
from scipy.sparse import csc_matrix, diags_array
from scipy.sparse.linalg import svds

def sparsify_matrix(A, eps_factor):
    '''
    Removes all values from A that are < eps.
    Use this function if a sparse matrix has elements stored in all entries,
    but many of these elements are numerically 0.

    Input
    -----
    A - scipy.sparse matrix
    eps - float
    '''
    eps = A.diagonal().mean() / eps_factor
    mask = np.abs(A.data) >= eps
    
    # Recompute indptr for the filtered data
    new_indptr = np.zeros(A.shape[1]+1, dtype=int)
    current_index = 0
    for col in range(A.shape[1]):
        start, end = A.indptr[col], A.indptr[col+1]
        # Count how many entries in this column are kept
        new_indptr[col+1] = new_indptr[col] + np.sum(mask[start:end])
    
    A_filt = csc_matrix((A.data[mask], A.indices[mask], new_indptr), shape=A.shape)

    # perc_occ = (A_filt.count_nonzero() / A_filt.size) * 100
    # if perc_occ > 30: # matrix more than 30 % occupied
    #     A_filt = A_filt.todense()
    #     print(f'Matrix is {perc_occ} % occupied. Converted to non-sparse data format.')
    return A_filt

def pseudoinverse(A):
    U, S, V = svds(A, k=A.shape[0]-1)
    S_inv = diags_array(1/S)
    A_inv = V.T @ S_inv @ U.T
    A_inv = csc_matrix(A)    
    return A_inv