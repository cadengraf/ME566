"""Camera-frame feature detection and matching experiments for homework 2."""
from pathlib import Path
from time import perf_counter
from importlib import import_module
from importlib.metadata import version

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .DoG import DoGFeatureDetector
from .FAST import FAST
from .Harris import HarrisFeatureDetector
from .SIFT import SIFT
from .ORB import ORB
from .BRIEF import BRIEF
from .ImageReader import read_images_from_folder
ShiTomasi = import_module(".Shi-Tomasi", __package__).ShiTomasi


METHODS = ("DoG", "Harris", "Shi-Tomasi", "FAST")
DESCRIPTORS = ("SIFT", "BRIEF", "ORB")


def load_sequences(root):
    """Read recorded camera frames in numeric order; no image transformation."""
    root = Path(root)
    image_root = root / "images_hw2" / "images"
    names = {"translation": "translation", "scale": "scale", "rotation": "rot"}
    return {motion: read_images_from_folder(image_root / f"hw2_{name}_images")
            for motion, name in names.items()}

def show_camera_views(sequences):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row, (motion, frames) in enumerate(sequences.items()):
        for ax, (name, image) in zip(axes[row],
                                     (frames[0], frames[len(frames) // 2], frames[-1])):
            ax.imshow(image, cmap="gray")
            ax.set_title(f"{motion}: {name}")
            ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_rotation_views(sequences, count=9):
    """Show evenly spaced recorded rotation frames without transforming them."""
    frames = sequences["rotation"]
    indices = np.unique(np.linspace(0, len(frames) - 1,
                                    min(count, len(frames)), dtype=int))
    columns = 3
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, index in zip(np.ravel(axes), indices):
        name, image = frames[index]
        ax.imshow(image, cmap="gray")
        ax.set_title(name)
    fig.suptitle("Camera rotation sequence (recorded frames)")
    fig.tight_layout()
    plt.show()


def show_detector_views(sequences, motion):
    """Compare all four detectors on the first, middle, and last camera views."""
    frames = sequences[motion]
    selected = (frames[0], frames[len(frames) // 2], frames[-1])
    fig, axes = plt.subplots(len(METHODS), 3, figsize=(12, 12))
    for col, (name, image) in enumerate(selected):
        for row, method in enumerate(METHODS):
            points, _ = detect(image, method)
            ax = axes[row, col]
            ax.imshow(image, cmap="gray")
            ax.scatter(points[:, 0], points[:, 1], s=3, c="lime",
                       linewidths=0, alpha=0.8)
            ax.set_title(f"{name} | {method}: {len(points)}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{motion.capitalize()} camera views with detections")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


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


def detect(gray, method):
    if method == "DoG":
        return DoGFeatureDetector(min_sigma=1, max_sigma=12,
                                  threshold=0.03).detect_points(gray)
    if method == "Harris":
        return HarrisFeatureDetector().detect_points(gray)
    if method == "Shi-Tomasi":
        return ShiTomasi(max_corners=2000).detect_points(gray)
    if method == "FAST":
        return FAST(threshold=20, nonmaxSupression=True).detect_points(gray)
    raise ValueError(method)


def show_detector_features(image):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, method in zip(axes, METHODS):
        points, _ = detect(image, method)
        ax.imshow(image, cmap="gray")
        ax.scatter(points[:, 0], points[:, 1], s=3, c="lime")
        ax.set_title(f"{method}: {len(points)}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def _occupied_cells(points, shape, cells=4):
    if not len(points):
        return 0
    h, w = shape
    xs = np.clip((points[:, 0] * cells / w).astype(int), 0, cells - 1)
    ys = np.clip((points[:, 1] * cells / h).astype(int), 0, cells - 1)
    return len(set(zip(xs, ys)))


def show_anms(image, keep=100):
    points, strengths = detect(image, "Harris")
    selected = anms(points, strengths, keep=keep)
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


def homography(a, b):
    sift = SIFT(nfeatures=3000)
    ka, da = sift.detect_and_compute(a)
    kb, db = sift.detect_and_compute(b)
    if da is None or db is None or len(db) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < 0.7 * n.distance]
    if len(good) < 4:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H if inliers is not None and inliers.sum() >= 10 else None


def repeatability(reference, target, H, shape, tolerance=5):
    if H is None or not len(reference) or not len(target):
        return np.nan
    projected = cv2.perspectiveTransform(reference.reshape(-1, 1, 2), H).reshape(-1, 2)
    h, w = shape
    visible = (np.isfinite(projected).all(axis=1)
               & (projected[:, 0] >= 0) & (projected[:, 0] < w)
               & (projected[:, 1] >= 0) & (projected[:, 1] < h))
    projected = projected[visible]
    if not len(projected):
        return np.nan
    hits = 0
    for chunk in np.array_split(projected, max(1, int(np.ceil(len(projected) / 256)))):
        hits += np.any(np.sum((chunk[:, None] - target[None]) ** 2, axis=2)
                       <= tolerance ** 2, axis=1).sum()
    return hits / len(projected)


def sampled_pairs(sequences):
    for motion, frames in sequences.items():
        sample = frames[1::5]
        if frames[-1][0] not in [name for name, _ in sample]:
            sample += [frames[-1]]
        for name, image in sample:
            yield motion, frames[0][1], name, image


def compare_detectors(sequences):
    records = []
    reference_points = {}
    for motion, frames in sequences.items():
        name, base = frames[0]
        for method in METHODS:
            start = perf_counter()
            points, _ = detect(base, method)
            ms = (perf_counter() - start) * 1000
            reference_points[motion, method] = points
            records.append((motion, name, method, len(points), ms, 1.0))
    for motion, base, name, image in sampled_pairs(sequences):
        H = homography(base, image)
        for method in METHODS:
            start = perf_counter()
            points, _ = detect(image, method)
            ms = (perf_counter() - start) * 1000
            score = repeatability(reference_points[motion, method],
                                  points, H, image.shape)
            records.append((motion, name, method, len(points), ms, score))
    return pd.DataFrame(records, columns=["motion", "frame", "detector",
                                          "features", "milliseconds", "repeatability"])


def show_detector_results(results):
    summary = results.groupby(["motion", "detector"])[
        ["features", "milliseconds", "repeatability"]].mean().round(3)
    print(summary.to_string())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex="row")
    for row, motion in enumerate(("translation", "scale", "rotation")):
        motion_results = results[results.motion == motion].copy()
        motion_results["frame_number"] = motion_results["frame"].str.extract(
            r"frame_(\d+)", expand=False).astype(int)
        for col, metric in enumerate(("features", "milliseconds", "repeatability")):
            ax = axes[row, col]
            for method in METHODS:
                series = motion_results[motion_results.detector == method].sort_values(
                    "frame_number")
                ax.plot(series.frame_number, series[metric], marker="o", label=method)
            ax.set_title(f"{motion}: {metric}")
            ax.set_xlabel("Recorded frame number")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def _keypoints(gray):
    points, strengths = detect(gray, "Harris")
    return [cv2.KeyPoint(float(x), float(y), 16)
            for x, y in anms(points, strengths, keep=300)]


def _descriptor(gray, kind):
    engines = {"SIFT": SIFT, "BRIEF": BRIEF, "ORB": ORB}
    if kind not in engines:
        raise ValueError(kind)
    return engines[kind]().compute(gray, _keypoints(gray))


def _match_labels(a, b, kind, H):
    start = perf_counter()
    ka, da = _descriptor(a, kind)
    kb, db = _descriptor(b, kind)
    if da is None or db is None or len(db) < 2:
        return np.array([]), np.array([], bool), (perf_counter() - start) * 1000
    norm = {"SIFT": SIFT, "BRIEF": BRIEF, "ORB": ORB}[kind].norm
    pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
    projected = cv2.perspectiveTransform(
        np.float32([point.pt for point in ka]).reshape(-1, 1, 2), H).reshape(-1, 2)
    ratios, labels = [], []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        ratios.append(m.distance / max(n.distance, 1e-12))
        labels.append(np.linalg.norm(projected[m.queryIdx] - kb[m.trainIdx].pt) <= 5)
    return np.asarray(ratios), np.asarray(labels, bool), (perf_counter() - start) * 1000


def compare_descriptors(sequences):
    records = []
    for motion, a, name, b in sampled_pairs(sequences):
        H = homography(a, b)
        if H is None:
            continue
        for kind in DESCRIPTORS:
            ratios, labels, ms = _match_labels(a, b, kind, H)
            if len(ratios):
                records.append((motion, kind, ratios, labels, ms))
    return records


def show_confusion_matrices(records, ratio_threshold=0.75):
    for kind in DESCRIPTORS:
        rows = [row for row in records if row[1] == kind]
        if not rows:
            print(f"{kind}: no valid matches")
            continue
        ratios = np.concatenate([row[2] for row in rows])
        labels = np.concatenate([row[3] for row in rows])
        matrix = confusion_matrix(labels, ratios < ratio_threshold,
                                  labels=[False, True])
        precision = matrix[1, 1] / max(matrix[:, 1].sum(), 1)
        print(f"{kind}: mean pair time {np.mean([row[4] for row in rows]):.2f} ms")
        print("Rows: actual wrong/right; columns: rejected/accepted")
        print(matrix)
        print(f"Accepted-match precision: {precision:.3f}\n")


def show_sift_roc(records):
    rows = [row for row in records if row[1] == "SIFT"]
    if not rows:
        print("ROC unavailable: no SIFT matches")
        return
    ratios = np.concatenate([row[2] for row in rows])
    labels = np.concatenate([row[3] for row in rows])
    if len(np.unique(labels)) < 2:
        print("ROC unavailable: only one geometric label class")
        return
    fpr, tpr, _ = roc_curve(labels, -ratios)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"SIFT AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.grid(True)
    plt.show()
