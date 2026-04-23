# Drift Happens

A tiny, presentation-ready machine learning demo inspired by **Google's Rules of ML**.

This project demonstrates two practical ideas in under 10 seconds:

1. A simple model can perform similarly to a more complex model.
2. Data drift can degrade model performance in production.

## Why this project exists

This is designed for a **live 5-minute demo**:
- Minimal code
- Deterministic output (fixed seeds)
- Clear, readable terminal messages
- No heavy dependencies

## Demos included

### Demo 1: Simple vs Complex Model

- Dataset: synthetic binary classification from `sklearn.datasets.make_classification`
- Models:
  - Logistic Regression (simple baseline)
  - Random Forest (complex model)
- Output:
  - Accuracy of both models
  - Accuracy gap
  - Short takeaway message

### Demo 2: Data Drift

- Train on Dataset A (original distribution)
- Evaluate on:
  - Dataset A (good performance)
  - Dataset B (drifted features: mean shift + scale change + noise)
- Output:
  - Performance before drift
  - Performance after drift
  - Performance drop and explanation

## Project structure

```text
.
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## How to run

1. Install Poetry (once):

```powershell
pip install poetry
```

2. Install dependencies:

```powershell
poetry install
```

3. Run the demo:

```powershell
poetry run python main.py
```

Optional (script entry point):

```powershell
poetry run drift-happens
```

## Alternative: pip + requirements.txt

If you prefer not to use Poetry, you can still run with `pip`:

1. Create and activate a virtual environment (recommended).

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the demo:

```powershell
python main.py
```

## Example output

```text
Drift Happens: Rules of ML Demo
This script demonstrates why simple baselines and drift monitoring matter.

======================================================================
DEMO 1: Simple vs Complex Model
======================================================================
Creating dataset and splitting into train/test...
Training Logistic Regression (simple baseline)...
Training Random Forest (more complex model)...

Results on the same test data:
Logistic Regression Accuracy: 0.840
Random Forest Accuracy:      0.867
Accuracy Gap (Complex - Simple): +0.027
Takeaway: The simple model performs similarly to the complex one.

======================================================================
DEMO 2: Data Drift in Production
======================================================================
Model trained on old data (Dataset A)...
Evaluating on familiar data distribution (Dataset A)...
Creating new data with drift (Dataset B)...
Evaluating the same model on drifted data (Dataset B)...

Drift Results:
Accuracy on Dataset A (old distribution): 0.840
Accuracy on Dataset B (with drift):       0.712
Performance Drop After Drift:             0.128
Explanation:
- Model trained on old data
- New data has drift
- Performance dropped

Done. Deterministic run complete (fixed random seed).
```

## How this relates to Google's Rules of ML

- **Rule: Start with simple baselines first.**
  Logistic Regression is fast, interpretable, and often surprisingly competitive.
- **Rule: Monitor your system in production.**
  Even a good model can fail when data distribution shifts.
- **Rule: Focus on reliability and iteration speed.**
  This demo favors clarity, determinism, and easy debugging over complexity.

## Suggested GitHub repository setup

Suggested public repository name:

- `drift-happens-rules-of-ml-demo`

### Initialize and push

```powershell
# From the project folder
cd "c:\Users\stefa\Desktop\New folder"

# Initialize git
git init

# Add files
git add main.py README.md requirements.txt

# Commit
git commit -m "Initial commit: Rules of ML demo (simple vs complex + data drift)"

# Create repo on GitHub, then connect remote
# Replace YOUR_USERNAME with your GitHub username
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/drift-happens-rules-of-ml-demo.git

# Push
git push -u origin main
```
