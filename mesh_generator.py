import numpy as np
import matplotlib.pyplot as plt

from skfem.visuals.matplotlib import draw
import cv2

from skfem import MeshTri


import numpy as np
import cv2
from scipy.spatial import Delaunay

def generate_mesh_from_mask(mask, max_points=500):
    # Extract contours
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask.")

    largest = max(contours, key=cv2.contourArea)
    pts = np.array(largest).reshape(-1, 2)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        raise ValueError("Too few points for mesh.")

    if len(pts) > max_points:
        pts = pts[np.random.choice(len(pts), max_points, replace=False)]

    # Normalize to unit square
    pts = pts.astype(np.float64)
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)

    tri = Delaunay(pts)
    if tri.simplices.shape[0] == 0:
        raise ValueError("Triangulation failed.")

    # Transpose to match skfem format
    return MeshTri(pts.T, tri.simplices.T)


def create_mesh_from_mask(mask, max_points=500):
    points = np.column_stack(np.where(mask > 0))

    if len(points) < 3:
        raise ValueError("Not enough ROI points to create mesh.")

    # Downsample
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    # Normalize points to [0, 1] square
    h, w = mask.shape
    points = points[:, [1, 0]] / np.array([w, h])  # Swap x/y and normalize

    # Create unit square mesh for visualization/demo (not real geometry yet)
    mesh = MeshTri.init_symmetric().refined(3)
    return mesh

def visualize_mesh(mesh):
    fig, ax = plt.subplots()
    draw(mesh, ax=ax)
    ax.set_title("Generated Mesh")
    plt.show()
