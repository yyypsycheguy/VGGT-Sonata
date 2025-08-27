import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Your tensor
tensor = torch.tensor([[[ 3.3458e-04],
         [ 2.9594e-04],
         [ 6.3570e-05]],

        [[ 7.8359e-01],
         [-1.2594e+00],
         [-2.1051e-01]],

        [[ 1.3741e+00],
         [-1.2425e+00],
         [-1.5034e-01]]])
print(f'tensor shape: {tensor.shape}')
# Reshape to (N, 3)
points = tensor.squeeze(-1)
#tensor[: ,1] *=3.1
print(f'tensor scaled; {tensor}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter points
ax.scatter(points[:, 1], points[:, 0], points[:, 2], c='red', s=50)

# Draw lines connecting points in order
ax.plot(points[:, 1], points[:, 0], points[:, 2], color='blue', linewidth=0.5)

# Add point indices only once per point
'''for i, (y, x, z) in enumerate(points):
    ax.text(y, x, z, f'{i}', color='black', fontsize=10)'''

ax.set_xlabel('Y towards') # switching the 2 axes for easier visualisation
ax.set_ylabel('X right')
ax.set_zlabel('Z up')
ax.set_title('Camera position in world coordinates ()')

plt.savefig('after-scale.png')
plt.show()