# Run the three-group and image-count calibration comparisons.
from pathlib import Path
import numpy as np
from .CalibrateCamera import CalibrateCamera

class CalibrationComparison:
    def __init__(self, image_dir=None):
        self.image_dir = Path(image_dir) if image_dir is not None else Path(__file__).resolve().parents[2] / "images/filtered_set_me566_1"

    def compare(self):
        # Inner corners; square_size=1 gives translations in chessboard-square units.
        camera1 = CalibrateCamera(image_dir=self.image_dir)

        # Split folder 1 into three disjoint groups for the data-set comparison.
        order1 = np.random.default_rng(0).permutation(len(camera1.imgpoints))
        groups = np.array_split(order1, 3)
        data_sets = [("Set 1A", camera1, groups[0]),
                     ("Set 1B", camera1, groups[1]),
                     ("Set 1C", camera1, groups[2])]

        results = {}
        for name, camera, indices in data_sets:
            rms, matrix, distortion, rvecs, tvecs = camera.calibrate_camera(indices=indices)
            results[name] = (rms, matrix, distortion, rvecs, tvecs)
            print(f"\n{name}: {len(indices)} images, RMS = {rms:.4f} pixels")
            print("Camera matrix:\n", matrix)
            print("Distortion [k1, k2, p1, p2, k3]:", distortion.ravel())

        # Nested, repeatable subsets: each larger trial keeps the previous images.
        print("\nSet 1: effect of image count (focal lengths and center in pixels)")
        print(" n    RMS        fx        fy        cx        cy     k1      k2      p1      p2      k3")
        for count in [2, 5, 10, 15, 25]:
            rms, matrix, distortion, rvecs, tvecs = camera1.calibrate_camera(count, order1)
            results[count] = (rms, matrix, distortion, rvecs, tvecs)
            print(f"{count:2d} {rms:7.4f} {matrix[0,0]:9.2f} {matrix[1,1]:9.2f} "
                  f"{matrix[0,2]:9.2f} {matrix[1,2]:9.2f} "
                  + " ".join(f"{value:7.3f}" for value in distortion.ravel()))
        return results

