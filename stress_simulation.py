import numpy as np
import matplotlib.pyplot as plt
from skfem import *
from skfem.helpers import dot, sym_grad, trace
from skfem.visuals.matplotlib import draw, plot
from scipy.spatial import Delaunay
import cv2


def compute_stress_distribution(mask, applied_force=10.0):
    # Extract the largest contour from the mask
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask.")

    largest_contour = max(contours, key=cv2.contourArea)
    pts = np.array(largest_contour).reshape(-1, 2)
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        raise ValueError("Contour is not valid or has fewer than 3 unique points.")

    # Downsample points to avoid degenerate triangulation
    if pts.shape[0] > 1000:
        pts = pts[np.random.choice(len(pts), 1000, replace=False)]

    # Normalize and triangulate
    pts = pts.astype(np.float64)
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)

    tri = Delaunay(pts)
    if tri.simplices.shape[0] == 0:
        raise ValueError("Triangulation failed: no triangles generated.")

    # Filter out degenerate triangles (area close to 0)
    p = pts.T
    t = tri.simplices.T
    a = np.linalg.norm(p[:, t[1]] - p[:, t[0]], axis=0)
    b = np.linalg.norm(p[:, t[2]] - p[:, t[1]], axis=0)
    c = np.linalg.norm(p[:, t[0]] - p[:, t[2]], axis=0)
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    valid = area > 1e-10
    t = t[:, valid]

    if t.shape[1] == 0:
        raise ValueError("All triangles were degenerate.")

    m = MeshTri(p, t)

    # Use vector elements for 2D elasticity
    element = ElementVector(ElementTriP1())
    basis = Basis(m, element)

    # Material properties
    E = 1e5
    nu = 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    @BilinearForm
    def a(u, v, w):
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        tr_eps_u = trace(eps_u)
        tr_eps_v = trace(eps_v)
        return lam * tr_eps_u * tr_eps_v + 2.0 * mu * dot(eps_u, eps_v)

    @LinearForm
    def l(v, w):
        return applied_force * v[1]

    A = asm(a, basis)
    b = asm(l, basis)

    # Pin bottom-most points
    ymin = m.p[1].min()
    fixed = np.where(np.abs(m.p[1] - ymin) < 1e-3)[0]
    D = basis.get_dofs(fixed, components=[0, 1])  # pin both x and y
    x = solve(*condense(A, b, D=D))

    # Interpolate displacement magnitude
    displacement = basis.project(np.linalg.norm(x.reshape(-1, 2), axis=1))
    return m, displacement


def visualize_stress(m, displacement):
    fig, ax = plt.subplots()
    draw(m, ax=ax, linewidth=0.3)
    plot(m, displacement, ax=ax, shading='gouraud')
    ax.set_title("Stress Distribution (Displacement)")
    plt.colorbar(ax.collections[0], ax=ax, label="Displacement")
    plt.show()
