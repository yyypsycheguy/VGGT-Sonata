import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Your tensor
tensor = torch.tensor([[[ 5.1080e-05],
         [-1.6961e-04],
         [-8.2121e-05]],

        [[-4.1403e-03],
         [-1.1220e-01],
         [ 2.6062e-03]],

        [[ 4.4396e-03],
         [-5.4602e-01],
         [ 1.6280e-02]],

        [[-1.6555e-03],
         [-1.1631e+00],
         [ 3.3659e-02]],

        [[-1.1139e-02],
         [-1.8261e+00],
         [ 5.6927e-02]],

        [[-1.6429e-02],
         [-2.4719e+00],
         [ 7.5969e-02]],

        [[-2.8063e-02],
         [-3.0306e+00],
         [ 9.3659e-02]],

        [[-4.8377e-02],
         [-3.4298e+00],
         [ 1.0913e-01]],

        [[-4.9002e-02],
         [-3.5478e+00],
         [ 1.1345e-01]]])
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
