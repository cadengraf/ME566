"""Adaptive nonmaximum suppression and its spatial coverage demonstration."""
import numpy as np
import matplotlib.pyplot as plt
from .Harris import HarrisFeatureDetector

def anms(points, strengths, keep=300, robust=0.9):
    """Retain points with the largest distance to a substantially stronger point."""
    points = np.asarray(points, np.float32).reshape(-1, 2)
    strengths = np.asarray(strengths, np.float32)
    if len(points) <= keep:
        return points
    radii = np.full(len(points), np.inf)
    for i in range(len(points)):
        stronger = strengths > strengths[i] / robust
        stronger[i] = False
        if np.any(stronger):
            radii[i] = np.min(np.sum((points[stronger] - points[i]) ** 2, axis=1))
    return points[np.argsort(radii)[-keep:]]


def _occupied_cells(points, shape, cells=4):
    if not len(points):
        return 0
    h, w = shape
    xs = np.clip((points[:, 0] * cells / w).astype(int), 0, cells - 1)
    ys = np.clip((points[:, 1] * cells / h).astype(int), 0, cells - 1)
    return len(set(zip(xs, ys)))


def show_anms(image, keep=100, robust=0.9):
    points, strengths = HarrisFeatureDetector().detect_points(image)
    selected = anms(points, strengths, keep=keep, robust=robust)
    strongest = points[np.argsort(strengths)[-len(selected):]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, chosen, title in zip(axes, (strongest, selected),
                                 ("Highest Harris responses", "Harris + ANMS")):
        ax.imshow(image, cmap="gray")
        ax.scatter(chosen[:, 0], chosen[:, 1], s=8, c="red")
        coverage = _occupied_cells(chosen, image.shape)
        ax.set_title(f"{title}: {len(chosen)} points, {coverage}/16 grid cells")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return selected


