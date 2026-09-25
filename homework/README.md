# Homework 1

The work is in [hw1.ipynb](hw1.ipynb), with supporting code in [utils/](utils/) and the writeup in [hw1-report.md](hw1-report.md). Input images are in [`../images/`](../images/).

## Linux setup

From the repository root, create a Python environment and install the shared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook homework/hw1.ipynb
```

Run the notebook cells in order from the repository root. Questions 1–4 can run independently; run question 4 before question 5 because question 5 uses its segmentation masks. Comparison figures are written to [`filter_results/`](filter_results/).

In VS Code, open `hw1.ipynb` and select the repository's `.venv` kernel. Run the cells in order.
