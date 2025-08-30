import matplotlib.pyplot as plt
import torch

"""Construct a 2D map with white background, green floor points, and red extrinsic points with labels"""

print(
    "################################# Construct 2D map ##################################\n"
)

if __name__ == "__main__":
    # Floor points example (replace with actual floor_coords)
    floor_coords = torch.load("scaled_floor_coords.pt")
    floor_coords = torch.tensor(floor_coords).cpu().numpy()
    name = torch.load("name.pt")

    # Extrinsics translation parts
    extrinsics = (
        torch.load("../vggt/extrinsic_scaled.pt").squeeze().cpu().numpy()
    )  # shape [S, 3]

    # Plot floor points in green
    plt.scatter(
        floor_coords[:, 0], floor_coords[:, 1], c="green", s=10, label="Floor Points"
    )

    # Plot extrinsics in red
    plt.scatter(
        extrinsics[:, 0], extrinsics[:, 1], c="red", s=50, label="Extrinsic Points"
    )

    # Axis labels
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Floor vs Extrinsic Positions (World Coordinates)")
    plt.legend()
    plt.axis("equal")
    plt.gca().invert_yaxis()

    plt.savefig("2Dmap_floor.png", dpi=300)
    plt.show()
    print("Saved enlarged floor map to 2Dmap_floor.png")
