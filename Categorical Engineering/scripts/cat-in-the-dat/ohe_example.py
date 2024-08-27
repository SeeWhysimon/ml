import numpy as np
from sklearn.preprocessing import OneHotEncoder

example = np.random.randint(1000, size=1000000)

ohe = OneHotEncoder(sparse_output=False)
ohe_example = ohe.fit_transform(example.reshape(-1, 1))
print(f'Size of dense array: {ohe_example.nbytes}')

ohe = OneHotEncoder(sparse_output=True)
ohe_example = ohe.fit_transform(example.reshape(-1, 1))
print(f'Size of sparse array: {ohe_example.data.nbytes}')

full_size = (ohe_example.data.nbytes + 
             ohe_example.indices.nbytes +
             ohe_example.indptr.nbytes)
print(f'Full size of sparse array: {full_size}')