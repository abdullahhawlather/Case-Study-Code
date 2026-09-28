import pandas as pd
from pathlib import Path

# ============================================================
# CUSTOMER SURVEY PROCESSING SCRIPT
# Recreates the structure of Customer_Survey_Final.xlsx
# from Survey_For_Prototype_(Responses)_Original.xlsx
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Survey_For_Prototype_(Responses)_Original.xlsx"
OUTPUT_FILE = BASE_DIR / "Customer_Survey_Final_Recreated.xlsx"

# -----------------------------
# 1. Read original survey data
# -----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=0)

# The original Google Forms export contains 17 columns.
CATEGORY_COL = df.columns[1]
FREQUENCY_COL = df.columns[2]
QUANTITY_COL = df.columns[3]
PRICE_SENSITIVITY_COL = df.columns[8]
PROMOTION_COL = df.columns[9]
SATISFACTION_COL = df.columns[11]
REPURCHASE_COL = df.columns[13]

# -----------------------------
# 2. Product Category -> Product ID
# -----------------------------
product_id_map = {
    "Electronics": "P001",
    "MRO(Maintenance, Repair and Operations)": "P002",
    "Office Supplies": "P003",
    "Packaging Materials": "P004",
    "Raw Materials": "P005",
}

# Short category names used in the processed file.
category_map = {
    "Electronics": "Electronics",
    "MRO(Maintenance, Repair and Operations)": "MRO",
    "Office Supplies": "Office Supplies",
    "Packaging Materials": "Packaging",
    "Raw Materials": "Raw Materials",
}

# -----------------------------
# 3. Purchase Frequency -> H/M/L
# -----------------------------
# High   = Weekly
# Medium = Monthly
# Low    = Quarterly / Bi-annually / Annually
frequency_map = {
    "Weekly": "High",
    "Monthly": "Medium",
    "Quarterly": "Low",
    "Bi-annually": "Low",
    "Annually": "Low",
}

# -----------------------------
# 4. Price Sensitivity -> H/M/L
# -----------------------------
price_sensitivity_map = {
    "Will continue buying the same quantity without hesitation": "Low",
    "Uncertain / Depends on market alternatives": "Medium",
    "Will reduce purchase volume / look for lower-cost alternatives": "High",
    "Will immediately switch to a competing supplier or product type": "High",
}

# -----------------------------
# 5. Promotion Response -> H/M/L
# -----------------------------
promotion_map = {
    "Do not influence at all": "Low",
    "Slightly influence (Nice to have, but doesn’t change core needs)": "Low",
    "Moderately influence (They encourage larger batch orders)": "Medium",
    "Strongly influence (I actively wait for promotions)": "High",
}

# -----------------------------
# 6. Satisfaction -> H/M/L
# -----------------------------
satisfaction_map = {
    "Very dissatisfied": "Low",
    "Dissatisfied": "Low",
    "Neutral": "Medium",
    "Satisfied": "High",
    "Very satisfied": "High",
}

# -----------------------------
# 7. Repurchase Intention -> H/M/L
# -----------------------------
repurchase_map = {
    "Extremely Unlikely": "Low",
    "Unlikely": "Low",
    "Neutral": "Medium",
    "Likely": "High",
    "Extremely Likely": "High",
}

# -----------------------------
# 8. Previous Purchase Quantity
# -----------------------------
def quantity_to_number(value):
    """Convert survey quantity ranges to a numeric estimate.

    Examples:
        1-100 units   -> 50.5
        101-500 units -> 300.5
        501-1000 units -> 750.5
        1000+ units   -> 1000.0 (lower bound)
        1500+         -> 1500.0 (lower bound)
        10-15         -> 12.5
        10-20         -> 15.0
        2000 units    -> 2000.0
        1-2 units     -> 1.5
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    text = text.replace("units", "").strip()

    # Exact number, e.g. 2000
    if text.replace(".", "", 1).isdigit():
        return float(text)

    # Open-ended response, e.g. 1000+ or 1500+
    if text.endswith("+"):
        number = text[:-1].strip()
        if number.replace(".", "", 1).isdigit():
            return float(number)

    # Range, e.g. 1-100, 101-500, 10-15
    parts = [p.strip() for p in text.split("-")]
    if len(parts) == 2:
        try:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2
        except ValueError:
            pass

    raise ValueError(f"Unrecognized quantity response: {value!r}")

# Standard quantity calculation.
previous_purchase = df[QUANTITY_COL].apply(quantity_to_number)

# ------------------------------------------------------------
# IMPORTANT:
# The existing Customer_Survey_Final.xlsx contains seven quantity
# values for C144-C150 that do NOT follow the raw survey responses.
# They are preserved below so the recreated output can match the
# existing processed file exactly.
#
# C144: raw = 101-500 units, normal midpoint = 300.5, existing = 206
# C145: raw = 1-100 units,    normal midpoint = 50.5,  existing = 206
# C146: raw = 1-100 units,    normal midpoint = 50.5,  existing = 205
# C147: raw = 1-100 units,    normal midpoint = 50.5,  existing = 204
# C148: raw = 1-100 units,    normal midpoint = 50.5,  existing = 205
# C149: raw = 1-100 units,    normal midpoint = 50.5,  existing = 206
# C150: raw = 1-100 units,    normal midpoint = 50.5,  existing = 208
# ------------------------------------------------------------
EXISTING_FILE_QUANTITY_OVERRIDES = {
    "C144": 206.0,
    "C145": 206.0,
    "C146": 205.0,
    "C147": 204.0,
    "C148": 205.0,
    "C149": 206.0,
    "C150": 208.0,
}

# -----------------------------
# 9. Build processed dataset
# -----------------------------
output = pd.DataFrame()

output["Consumer"] = [f"C{i:03d}" for i in range(1, len(df) + 1)]
output["Product ID"] = df[CATEGORY_COL].map(product_id_map)
output["Product_Category"] = df[CATEGORY_COL].map(category_map)
output["Purchase Frequency"] = df[FREQUENCY_COL].map(frequency_map)
output["Previous Purchase"] = previous_purchase
output["Price Sensitivity"] = df[PRICE_SENSITIVITY_COL].map(price_sensitivity_map)
output["Promotion Response"] = df[PROMOTION_COL].map(promotion_map)
output["Satisfaction"] = df[SATISFACTION_COL].map(satisfaction_map)
output["Repurchase Intention"] = df[REPURCHASE_COL].map(repurchase_map)
output["Represents"] = 1

# Apply the seven historical overrides only when reproducing the
# existing Customer_Survey_Final.xlsx.  For a clean methodology-
# based dataset, remove/comment out this section.
for consumer_id, value in EXISTING_FILE_QUANTITY_OVERRIDES.items():
    row = output["Consumer"] == consumer_id
    output.loc[row, "Previous Purchase"] = value

# Validate that every response was successfully mapped.
required_columns = [
    "Product ID",
    "Product_Category",
    "Purchase Frequency",
    "Previous Purchase",
    "Price Sensitivity",
    "Promotion Response",
    "Satisfaction",
    "Repurchase Intention",
]

for col in required_columns:
    if output[col].isna().any():
        bad_rows = output.index[output[col].isna()].tolist()
        raise ValueError(
            f"Unmapped values found in '{col}' at source rows: {bad_rows}. "
            "Check whether the survey introduced a new response option."
        )

# -----------------------------
# 10. Save final processed file
# -----------------------------
output.to_excel(OUTPUT_FILE, index=False, sheet_name="Sheet1")

print(f"Processing complete: {OUTPUT_FILE}")
print(f"Rows processed: {len(output)}")
print("Columns:", list(output.columns))
