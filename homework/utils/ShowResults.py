# Display and save the comparison figures for questions 2-5.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_path = Path(__file__).resolve().parents[1] / "filter_results"
def show_results(images, titles, filename):
    output_path.mkdir(exist_ok=True)
    columns = min(4, len(images))
    fig, axes = plt.subplots(int(np.ceil(len(images) / columns)), columns,
                             figsize=(4 * columns, 4 * int(np.ceil(len(images) / columns))))
    for ax in np.asarray(axes).ravel():
        ax.axis("off")
    for ax, image, title in zip(np.asarray(axes).ravel(), images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
    fig.suptitle(filename)
    fig.tight_layout()
    fig.savefig(output_path / (filename + ".png"))
    plt.show()
    plt.close(fig)


