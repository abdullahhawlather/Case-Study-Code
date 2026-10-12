"""Run all analysis scripts in dependency order.

The legacy script name 02_lstm_seed_stability.py is retained, but that script now
produces deterministic Holt damped-trend forecast diagnostics; it does not use LSTM.
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
scripts = [
    "00_reproduce_prototype_outputs.py",
    "01_survey_analysis.py",
    "02_lstm_seed_stability.py",
    "03_forecast_evaluation.py",
    "04_delay_model.py",
    "05_mcdm_comparison.py",
    "06_seasonal_factors.py",
    "07_supplier_table.py",
    "08_scenario_analysis.py",
]
for script in scripts:
    print(f"\n=== running {script} ===", flush=True)
    subprocess.run([sys.executable, str(here / script)], check=True, cwd=here)
