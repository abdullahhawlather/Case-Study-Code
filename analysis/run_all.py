"""Run every analysis script in order (the two LSTM scripts take several minutes)."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for script in ["00_reproduce_prototype_outputs.py", "01_survey_analysis.py", "02_lstm_seed_stability.py",
               "03_forecast_evaluation.py", "04_delay_model.py", "05_mcdm_comparison.py",
               "06_seasonal_factors.py", "07_supplier_table.py", "08_scenario_analysis.py"]:
    print("=== running", script)
    subprocess.run([sys.executable, str(here / script)], check=True, cwd=here)
