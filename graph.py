import matplotlib.pyplot as plt
import numpy as np

matrix = np.array([
    [-1.e-05, -1.e-05,  4.e-05],
    [ 0.00108,  0.05701, -0.68957],
    [ 0.00186,  0.02651, -0.33061],
    [-0.00954, -0.02900,  0.33588],
    [-0.00818,  0.04914, -0.61511]
]) # this matrix excludes first and last row, yields smoother results?

# Transpose so that each column is a row (for plotting)
matrix_T = matrix.T 

# Plot each column as a line
for i, col in enumerate(matrix_T):
    plt.plot(range(len(col)), col, label=f'Column {i}')

plt.title('Line Graph of Matrix Columns')
plt.xlabel('Row Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('xyz4384_mycode.png')
plt.show()
