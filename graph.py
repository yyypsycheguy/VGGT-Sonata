import matplotlib.pyplot as plt
import numpy as np

# Your 5x3 matrix
matrix = np.array([
    [ 0.01707,  0.07453, -1.35962],
    [ 0.01902,  0.04899, -1.06087],
    [ 0.01043,  0.00286, -0.49888],
    [ 0.01319, -0.02319, -0.21678],
    [ 0.01102, -0.03625, -0.03826]])

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
plt.savefig('matrix_columns_plot.png')
plt.show()
