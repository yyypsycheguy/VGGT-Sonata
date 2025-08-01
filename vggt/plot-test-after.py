import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Your tensor
tensor = torch.tensor([[[ 2.6234e-04],
         [-1.6961e-04],
         [-4.2177e-04]],

        [[-2.1265e-02],
         [-1.1220e-01],
         [ 1.3385e-02]],

        [[ 2.2802e-02],
         [-5.4602e-01],
         [ 8.3614e-02]],

        [[-8.5025e-03],
         [-1.1631e+00],
         [ 1.7287e-01]],

        [[-5.7208e-02],
         [-1.8261e+00],
         [ 2.9238e-01]],

        [[-8.4379e-02],
         [-2.4719e+00],
         [ 3.9017e-01]],

        [[-1.4413e-01],
         [-3.0306e+00],
         [ 4.8103e-01]],

        [[-2.4846e-01],
         [-3.4298e+00],
         [ 5.6049e-01]],

        [[-2.5167e-01],
         [-3.5478e+00],
         [ 5.8267e-01]]])
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
ax.set_title('Camera position in world coordinates')

plt.savefig('after-scale.png')
plt.show()
