# Analysis scripts

Code that reproduces the tables and figures of the case study **"An AI-Driven Dynamic Supplier Selection Framework with Hybrid Demand Forecasting and Scenario-Based Decision Support"** (Abdullah Hawlather, Md. Parvez Khan).

Place this folder (`analysis/`) in the root of the repository, next to `Logic.py` and `App.py`.

## Setup
```
pip install -r requirements.txt
```
Put `Survey_Responses_Anonymised.xlsx` in `Prototype Upload Data/Main Dataset/`. The scripts use it if present and never read respondent names.

## Run
```
python run_all.py            # all scripts (the two LSTM scripts take several minutes)
python 02_lstm_seed_stability.py --seeds 10
python 03_forecast_evaluation.py --seeds 5
```
Results are written to `results/` (JSON) and `figures/` (PNG). LSTM results vary slightly between runs because of random initialisation.

## Script → paper map (numbering of the final paper)
| Script | Paper section | Output |
|---|---|---|
| `00_reproduce_prototype_outputs.py` | 9.5, 9.6 | D_h, D_c, feasibility and ranking per product / combined view; original additive hybrid |
| `01_survey_analysis.py` | 2.2, 3.3 | Figures 5–11, Table 6 (Spearman), Table 7 (chi-square), Table 8 (g_c); collapsed chi-square + Monte Carlo (supplementary) |
| `02_lstm_seed_stability.py` | 9.5 | Table 11 (LSTM over random seeds) |
| `03_forecast_evaluation.py` | 10.1 | Table 14 (forecast errors vs baselines) |
| `04_delay_model.py` | 4, 10.2 | Table 15 (delay models, leakage check, threshold sensitivity) |
| `05_mcdm_comparison.py` | 10.3 | Supplier-rank table (SAW / TOPSIS / entropy), Kendall tau, weight Monte Carlo |
| `06_seasonal_factors.py` | 2.1 | Table 4 and Figure 2 (seasonal factors) |
| `07_supplier_table.py` | 2.1 | Table 3 (supplier information), derived from orders and compared with `Supplier Data.xlsx` |
| `08_scenario_analysis.py` | 9.5, 9.6 | Table 12 (ranking and feasibility) and Table 13 (demand scenarios); run after 01 and 02 |
`common.py` holds the shared paths and the LSTM configuration used by the prototype (paper Section 9.3).

## Data
Purchase orders: Kabir, S. (2024). Procurement KPI Analysis Dataset, Kaggle. Survey: 150 responses collected 24–26 September 2026 (anonymised).
