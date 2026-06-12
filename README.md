# Sustainable HR Tool

Small project for exploring and modelling sustainable heart rate from running data.

Quick start

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Generate dummy data (optional) and run the analysis script:

```powershell
python -c "from src import utils; utils.generate_dummy_data()"
python main.py
```

3. Open `notebooks/single_run_demo.ipynb` for interactive exploration.

Project layout

- `src/` — Python modules
- `data/` — raw and processed CSVs
- `notebooks/` — exploratory notebooks

