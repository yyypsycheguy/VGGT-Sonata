import torch
import matplotlib.pyplot as plt

# t-extrinsic unscaled
tensor = torch.tensor([[-1.0819e-05,  3.6083e-05, -1.3840e-05],
                        [-6.2035e-02, -1.2410e-01,  9.4065e-01],
                        [-1.5010e-02, -7.0181e-02,  3.7236e-01],
                        [-1.3972e-02, -1.2357e-01,  7.7372e-01]])

# Plot each axis
plt.plot(tensor[:, 0], label='X-axis')
plt.plot(tensor[:, 1], label='Y-axis')
plt.plot(tensor[:, 2], label='Z-axis')

plt.xlabel('Point Index')
plt.ylabel('Value')
plt.title('Axis Values Across Points')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('before_scale_plot.png')
plt.show()