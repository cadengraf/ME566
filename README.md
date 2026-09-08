# ME566 - Homework setup

Run these commands in PowerShell from the repository root. Keep `homework/` and `images/` in their current locations.

1. Create a virtual environment if `.venv` does not already exist:

```powershell
python3 -m venv .venv
```

2. Activate it and install the dependencies:

```powershell
source .venv/bin/activate
pip install -r requirements.txt
```

3. Open the notebook:

```powershell
python -m notebook homework/hw1.ipynb
jupyter server list

# Ctrl + click on the link with  the token
# Currently running servers (EXAMPLE):
http://localhost:8888/?token=???????
```

Alternatively, open `homework/hw1.ipynb` in VS Code and select the `.venv` Python environment using **Select Kernel**.

Questions 1-4 can run independently. Run question 4 before question 5 because it creates the segmentation masks. To run the full assignment, execute the question cells in order and save the notebook with its outputs. Comparison figures are saved in `homework/filter_results/`.

The writeup, equations, results and AI-use disclosure are in [hw1-report.md](homework/hw1-report.md).
