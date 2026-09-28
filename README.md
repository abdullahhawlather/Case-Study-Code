# Supplier Decision Console — Streamlit prototype

Python rebuild of the same tool, using the same formulas as the HTML/JS version.

## Run it

```bash
pip install -r requirements.txt
streamlit run App.py
```

Opens at http://localhost:8501 — pre-loaded with the example data from the proposal.

## Files

- 'App.py` — the Streamlit UI (tables, sliders, upload widgets, layout)
- `Logic.py` — all the math, kept separate from the UI so it can be tested or reused on its own
  (consumer-demand heuristic, trend forecast, feasibility check, supplier scoring, weight
  redistribution, xlsx/csv import/export)
- `.streamlit/config.toml` — dark theme with an amber accent, matching the original console
- `requirements.txt` — the four packages it needs

## Notes

- Upload buttons accept `.xlsx`, `.xls`, or `.csv` for each of the three data tables.
  Supplier files can be laid out either as one row per supplier, or as criteria down the first
  column with suppliers across the top (the layout used in the proposal's Section 9E) — the
  importer detects which one it's looking at.
- The consumer-demand and historical-trend formulas are simplified stand-ins for the XGBoost
  and LSTM models described in the proposal — see the footnote under each section in the app,
  or the docstrings in `Logic.py`, for exactly what they compute.
- The five decision weights always sum to 100%; moving one redistributes the rest proportionally.
