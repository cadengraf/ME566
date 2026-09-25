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
from .ANMS import anms
from .ImageReader import read_images_from_folder
ShiTomasi = import_module(".Shi-Tomasi", __package__).ShiTomasi


METHODS = ("DoG", "Harris", "Shi-Tomasi", "FAST")
DESCRIPTORS = ("SIFT", "BRIEF", "ORB")

# Edit these through comparison.SETTINGS in the notebook, then rerun the result cells.
SETTINGS = {
    "dog_threshold": 0.03,
    "harris_threshold": 0.01,
    "shi_quality": 0.01,
    "fast_threshold": 20,
    "registration_ratio": 0.7,
    "ransac_pixels": 3.0,
    "repeatability_pixels": 5.0,
    "geometric_match_pixels": 5.0,
    "match_ratio": 0.75,
    "anms_keep": 300,
    "anms_robust": 0.9,
}



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


def detect(gray, method):
    if method == "DoG":
        return DoGFeatureDetector(min_sigma=1, max_sigma=12,
                                  threshold=SETTINGS["dog_threshold"]).detect_points(gray)
    if method == "Harris":
        return HarrisFeatureDetector().detect_points(gray, threshold=SETTINGS["harris_threshold"])
    if method == "Shi-Tomasi":
        return ShiTomasi(max_corners=2000, quality_level=SETTINGS["shi_quality"]).detect_points(gray)
    if method == "FAST":
        return FAST(threshold=SETTINGS["fast_threshold"], nonmaxSupression=True).detect_points(gray)
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


def homography(a, b):
    sift = SIFT(nfeatures=3000)
    ka, da = sift.detect_and_compute(a)
    kb, db = sift.detect_and_compute(b)
    if da is None or db is None or len(db) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < SETTINGS["registration_ratio"] * n.distance]
    if len(good) < 4:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, SETTINGS["ransac_pixels"])
    return H if inliers is not None and inliers.sum() >= 10 else None


def repeatability(reference, target, H, shape, tolerance=None):
    if tolerance is None:
        tolerance = SETTINGS["repeatability_pixels"]
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
            for x, y in anms(points, strengths, keep=SETTINGS["anms_keep"],
                             robust=SETTINGS["anms_robust"])]


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
        labels.append(np.linalg.norm(projected[m.queryIdx] - kb[m.trainIdx].pt) <= SETTINGS["geometric_match_pixels"])
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


def show_confusion_matrices(records, ratio_threshold=None):
    if ratio_threshold is None:
        ratio_threshold = SETTINGS["match_ratio"]
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


def show_descriptor_visualization(image):
    """Display common Harris points and one descriptor vector from each method."""
    keypoints = _keypoints(image)
    fig, axes = plt.subplots(2, 3, figsize=(16, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    for col, kind in enumerate(DESCRIPTORS):
        kp, desc = {"SIFT": SIFT, "BRIEF": BRIEF, "ORB": ORB}[kind]().compute(
            image, keypoints)
        ax = axes[0, col]
        ax.imshow(image, cmap="gray")
        if kp:
            xy = np.asarray([point.pt for point in kp])
            ax.scatter(xy[:, 0], xy[:, 1], s=9, facecolors="none",
                       edgecolors="#00e5ff", linewidths=0.5)
            chosen = len(kp) // 2
            x, y = kp[chosen].pt
            ax.scatter([x], [y], s=100, facecolors="none",
                       edgecolors="yellow", linewidths=2)
        ax.set_title(f"{kind}: {len(kp)} descriptors")
        ax.axis("off")
        vector_ax = axes[1, col]
        if desc is not None and len(desc):
            vector = desc[chosen]
            if kind == "SIFT":
                vector_ax.imshow(vector.reshape(8, 16), cmap="magma", aspect="auto")
                vector_ax.set_title("Selected 128 float values (8 × 16)")
            else:
                bits = np.unpackbits(vector).reshape(16, 16)
                vector_ax.imshow(bits, cmap="gray_r", vmin=0, vmax=1,
                                 interpolation="nearest", aspect="auto")
                vector_ax.set_title("Selected 256 binary tests (16 × 16)")
            vector_ax.set_xticks([])
            vector_ax.set_yticks([])
    fig.suptitle("One Harris detection scheme + ANMS; yellow = displayed descriptor")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_match_visualization(a, b, H, ratio_threshold=None, limit=30):
    """Show accepted correspondences, green for geometrically right and red for wrong."""
    if ratio_threshold is None:
        ratio_threshold = SETTINGS["match_ratio"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))
    for ax, kind in zip(axes, DESCRIPTORS):
        ka, da = _descriptor(a, kind)
        kb, db = _descriptor(b, kind)
        canvas = np.concatenate((a, b), axis=1)
        ax.imshow(canvas, cmap="gray")
        if da is not None and db is not None and len(db) >= 2:
            norm = {"SIFT": SIFT, "BRIEF": BRIEF, "ORB": ORB}[kind].norm
            pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
            accepted = [(m, m.distance / max(n.distance, 1e-12))
                        for pair in pairs if len(pair) == 2
                        for m, n in [pair] if m.distance < ratio_threshold * n.distance]
            accepted.sort(key=lambda item: item[1])
            projected = (cv2.perspectiveTransform(
                np.float32([point.pt for point in ka]).reshape(-1, 1, 2), H
            ).reshape(-1, 2) if H is not None else None)
            correct = 0
            for m, _ in accepted[:limit]:
                x1, y1 = ka[m.queryIdx].pt
                x2, y2 = kb[m.trainIdx].pt
                right = (projected is not None and
                         np.linalg.norm(projected[m.queryIdx] - (x2, y2)) <= SETTINGS["geometric_match_pixels"])
                correct += right
                color = "#52e080" if right else "#ff5b5b"
                ax.plot((x1, x2 + a.shape[1]), (y1, y2), color=color,
                        alpha=0.8, linewidth=0.8)
            ax.set_title(f"{kind}: {len(accepted)} accepted; displayed {min(limit, len(accepted))} "
                         f"best ratios ({correct} geometrically right)")
        ax.axis("off")
    fig.suptitle(f"Ratio < {ratio_threshold:g} | green: within {SETTINGS['geometric_match_pixels']:g} px of homography")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_descriptor_summary(records, ratio_threshold=None):
    if ratio_threshold is None:
        ratio_threshold = SETTINGS["match_ratio"]
    """Plot matching speed and quality next to three labeled confusion matrices."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, kind in enumerate(DESCRIPTORS):
        rows = [row for row in records if row[1] == kind]
        if not rows:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue
        ratios = np.concatenate([row[2] for row in rows])
        labels = np.concatenate([row[3] for row in rows])
        matrix = confusion_matrix(labels, ratios < ratio_threshold,
                                  labels=[False, True])
        ax = axes[0, col]
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black", fontsize=15)
        ax.set_xticks((0, 1), ("Rejected", "Accepted"))
        ax.set_yticks((0, 1), ("Wrong", "Right"))
        ax.set_xlabel("Ratio-test decision")
        ax.set_ylabel("Geometric label")
        ax.set_title(f"{kind} confusion matrix")
        precision = matrix[1, 1] / max(matrix[:, 1].sum(), 1)
        recall = matrix[1, 1] / max(matrix[1].sum(), 1)
        mean_ms = np.mean([row[4] for row in rows])
        ax = axes[1, col]
        ax.bar(("Precision", "Recall"), (precision, recall),
               color=("#2868a2", "#4ba881"))
        ax.set_ylim(0, 1)
        ax.set_title(f"{mean_ms:.1f} ms/pair | {len(labels)} candidates")
        for i, value in enumerate((precision, recall)):
            ax.text(i, value + 0.02, f"{value:.2f}", ha="center")
    fig.suptitle(f"Shared Harris + ANMS detections | ratio threshold {ratio_threshold:g}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
