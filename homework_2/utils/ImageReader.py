"""Load the camera frames for homework 2 in numeric frame order."""
from pathlib import Path
import cv2


def read_images_from_folder(folder, grayscale=True):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths = sorted(folder.glob("frame_*.jpg"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise ValueError(f"No frame_*.jpg images in {folder}")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    frames = []
    for path in paths:
        image = cv2.imread(str(path), flag)
        if image is None:
            raise ValueError(f"Could not decode {path}")
        frames.append((path.name, image))
    return frames
