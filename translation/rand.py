import numpy as np
import sys

def generate_matrix(n, m):
    # Create an empty matrix of size n*8
    matrix = np.empty((n, 8), dtype=int)

    # Iterate through each row
    for i in range(n):
        # Generate a random permutation of integers from 0 to m-1
        row_values = np.random.permutation(m)[:8]
        matrix[i] = row_values

    return matrix

# Example: Generate a matrix of size 5*8 with each entry sampled from 0 to 10-1 without repetition
n = 99
m = int(sys.argv[2])
result_matrix = generate_matrix(n, m)

# Save the matrix to a .npy file
np.save('data/new/{}/rand.npy'.format(sys.argv[1]), result_matrix)
