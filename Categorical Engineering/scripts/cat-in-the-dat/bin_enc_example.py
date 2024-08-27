import numpy as np
from scipy import sparse

n_rows = 10000
n_cols = 100000

example = np.random.binomial(1, 0.05, size=(n_rows, n_cols))
print(f'Size of dense array: {example.nbytes}')

sparse_example = sparse.csr_matrix(example)
print(f'Size of sparse array: {sparse_example.data.nbytes}')

full_size = (sparse_example.data.nbytes + 
             sparse_example.indices.nbytes + 
             sparse_example.indptr.nbytes)
print(f'Full size of sparse array: {full_size}')