# Homework 2

The work is in [hw2.ipynb](hw2.ipynb), with supporting code in [utils/](utils/) and the writeup in [hw2-report.md](hw2-report.md). Input images are in [`../images_hw2/`](../images_hw2/).

## Linux setup

From the repository root, create a Python environment and install the shared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook homework_2/hw2.ipynb
```

Run the notebook cells from top to bottom. Change thresholds in the settings cell near the top, then rerun that cell and the result cells to update the figures and tables.

In VS Code, open `hw2.ipynb` and select the repository's `.venv` kernel. Run the cells in order.
