import matplotlib.pyplot as plt

import torch

# Your track_list
track_list = [
    torch.tensor([[[[147.0000, 259.0000]],
                   [[182.1812, 177.1020]],
                   [[161.9500, 196.7987]],
                   [[192.9694, 173.2325]]]], device='cuda:0'),
    torch.tensor([[[[147.0000, 259.0000]],
                   [[180.4837, 178.4607]],
                   [[162.1420, 196.7266]],
                   [[191.8119, 173.8730]]]], device='cuda:0'),
    torch.tensor([[[[147.0000, 259.0000]],
                   [[180.0810, 178.8796]],
                   [[162.2330, 196.8655]],
                   [[191.5559, 174.1803]]]], device='cuda:0'),
    torch.tensor([[[[147.0000, 259.0000]],
                   [[179.9152, 178.9544]],
                   [[162.2668, 196.8687]],
                   [[191.3947, 174.2044]]]], device='cuda:0')
]

# Stack them → shape = (4, 1, 4, 1, 2)
merged = torch.stack(track_list)  

# Squeeze to remove unnecessary dims → (4, 4, 2)
merged = merged.squeeze(1).squeeze(2)

print("Merged shape:", merged.shape)
# Output: (4, 4, 2)

# Example: use the first query point's track
tracked_points = track_list[0][0, :, 0, :].cpu().numpy()  # shape (4, 2)

fig, axes = plt.subplots(1, tracked_points.shape[0], figsize=(15, 5))

for i, ax in enumerate(axes):
    img = images[0, i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    ax.imshow(img.astype('uint8'))
    x, y = tracked_points[i]
    ax.plot(x, y, 'ro', markersize=5)  # tracked point
    ax.set_title(f"Frame {i}")
    ax.axis('off')

p,t.savefig('tracked_points.png')
plt.show()
