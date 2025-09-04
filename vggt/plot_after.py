import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Your tensor
tensor = torch.tensor([[[ 8.4420e-05],
         [ 3.1981e-04],
         [-1.1204e-03]],

        [[-4.6134e+00],
         [ 6.7566e-01],
         [ 5.7859e-01]],

        [[-3.9539e+00],
         [ 6.2482e-01],
         [ 5.0376e-01]],

        [[-3.4487e+00],
         [ 2.9517e-01],
         [ 4.1012e-01]],

        [[-3.0361e+00],
         [ 7.7288e-01],
         [ 4.2581e-01]],

        [[-2.3382e+00],
         [ 7.0531e-01],
         [ 3.3158e-01]],

        [[-1.4084e+00],
         [ 7.2093e-01],
         [ 1.9664e-01]],

        [[-5.0310e-01],
         [ 3.1103e-01],
         [ 7.7370e-02]],

        [[ 8.0916e-02],
         [ 1.7385e-01],
         [ 9.2119e-04]],

        [[ 8.7856e-02],
         [-8.2959e-03],
         [-4.8885e-02]],

        [[ 9.4015e-02],
         [-1.4552e-01],
         [-3.9198e-02]],

        [[ 1.0096e-01],
         [-5.2029e-02],
         [ 5.2665e-03]],

        [[-1.6036e-02],
         [-4.1343e-02],
         [-6.4365e-03]],

        [[ 9.9715e-02],
         [-6.4769e-02],
         [-4.0875e-02]]])
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