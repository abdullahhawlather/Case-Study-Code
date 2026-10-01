"""Supplier-level table derived from the purchase orders, compared with the static file.

Paper link: Section 2.1 - Table 2 (field definitions) and Table 3 (supplier information).
Prints the table computed from Purchase_Orders.xlsx by Logic.derive_supplier_table_from_raw
next to the values stored in 'Prototype Upload Data/Supplier Data.xlsx', so that the Capacity
column used in the paper can be traced to one source.
Output: results/supplier_table.json
"""
import pandas as pd
import common  # noqa: F401  (adds the repository root to sys.path)
import Logic as L
from common import DATA_DIR, load_orders, save_json

derived = L.derive_supplier_table_from_raw(load_orders(), None)
static = pd.read_excel(DATA_DIR / "Supplier Data.xlsx")
cols = ["Supplier", "Unit Price", "Lead Time (d)", "Quality %", "Capacity", "Delay Rate %",
        "Defect Rate %", "Reliability %", "Transport Cost"]
print("Derived from purchase orders (Capacity = total quantity / 90 x U(0.8, 1.3), seed 7):")
print(derived[cols].round(2).to_string(index=False))
print("\\nStatic file 'Supplier Data.xlsx':")
print(static.round(2).to_string(index=False))
save_json({"derived_from_orders": derived[cols].round(2).to_dict("records"),
           "static_file": static.round(2).to_dict("records")}, "supplier_table.json")
