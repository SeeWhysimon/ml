import numpy as np
import config

def add_matrices(matrix1, matrix2):
    return np.add(matrix1, matrix2)

if __name__ == "__main__":
    matrix1 = np.array([[1, 2], 
                        [3, 4]])
    matrix2 = np.array([[3, 2], 
                        [5, 4]])
    result = add_matrices(matrix1, matrix2)
    print(result)