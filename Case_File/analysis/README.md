# Analysis scripts

Code for the case study **“An AI-Driven Dynamic Supplier Selection Framework with Hybrid Demand Forecasting and Scenario-Based Decision Support.”** The historical-demand time-series model is **Holt's damped-trend exponential smoothing**; LSTM is no longer used by this analysis workflow.

Place these scripts beside `Logic.py` and `App.py`, or inside an `analysis/` folder at the repository root. `common.py` supports both the original `Prototype Upload Data/` layout and the flat folder layout used for the uploaded files.

## Setup

```bash
pip install -r requirements.txt
```

Required workbooks are `Purchase_Orders.xlsx`, `Historical Data.xlsx`, `Customer_Survey_Final.xlsx`, and `Supplier Data.xlsx`. If available, `Survey_Responses_Anonymised.xlsx` is preferred for survey analysis; otherwise the script uses `Survey_For_Prototype_(Responses)_Original.xlsx` and excludes name/respondent columns.

## Run

```bash
python run_all.py
```

The scripts write JSON results under `results/` and plots under `figures/`. The legacy filename `02_lstm_seed_stability.py` is retained to avoid breaking existing project references, but it now computes deterministic Holt damped-trend forecasts and writes `results/holt_forecast_diagnostics.json`. It does not train an LSTM and does not report random-seed variability.

## Script map

| Script | Purpose | Output / dependency |
|---|---|---|
| `00_reproduce_prototype_outputs.py` | Reproduce prototype demand and supplier rankings using the forecast function in `Logic.py` | `prototype_outputs.json` |
| `01_survey_analysis.py` | Survey statistics, correlations, tests, segments, and survey-implied growth | `survey_results.json`, figures |
| `02_lstm_seed_stability.py` | Legacy filename; Holt damped-trend 30-day forecast diagnostics by product and combined series | `holt_forecast_diagnostics.json` |
| `03_forecast_evaluation.py` | Chronological rolling one-step evaluation of naive, moving-average, mean, SES, Holt damped trend, and ARIMA models | `forecast_evaluation.json` |
| `04_delay_model.py` | Supplier delivery-delay model comparison and leakage checks | `delay_model.json` |
| `05_mcdm_comparison.py` | Compare prototype ranking, SAW, TOPSIS, entropy weights, and weight sensitivity | `mcdm_comparison.json` |
| `06_seasonal_factors.py` | Monthly seasonal factors using the ratio-to-average method | `seasonal_factors.json`, figure |
| `07_supplier_table.py` | Compare derived supplier metrics with the static supplier workbook | `supplier_table.json` |
| `08_scenario_analysis.py` | Apply low, normal, and high demand scenarios to supplier capacity and ranking | `scenario_analysis.json`; requires outputs from scripts 01 and 02 |
| `common.py` | Shared file paths, workbook loaders, JSON output, Holt forecast helper | Imported by scripts |
| `hybrid_patch.py` | Survey growth and revised hybrid-demand formula | Imported by scenario analysis |
| `run_all.py` | Execute the analysis in dependency order | Runs scripts above |

## Forecasting method

The historical demand forecast uses Holt's damped-trend exponential smoothing. The model is deterministic for a fixed input series and fitted configuration, so seed-stability analysis is not appropriate. Forecast evaluation uses chronological splits and rolling one-step-ahead predictions to reduce look-ahead leakage.

The scenario analysis applies the revised hybrid formula:

`D_hybrid = D_h * (1 + alpha * g_c) * scenario`

where `D_h` is the Holt historical forecast and `g_c` is survey-implied relative demand growth. The prototype reproduction script retains the additive hybrid implemented in `Logic.py` so that its output continues to reproduce the current application behaviour.
