import numpy as np

def rank_mean(probas: np.array):
    sorted = np.argsort(probas, axis=1)
    print(sorted)
    result = np.mean(sorted, axis=0)
    return result

if __name__ == "__main__":
    input_matrix = np.array([[0.9, 0.6, 0.3],
                             [0.8, 0.5, 0.2],
                             [0.7, 0.4, 0.1]])
    output_matrix = rank_mean(input_matrix)
    print(output_matrix)