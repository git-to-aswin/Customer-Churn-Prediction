"""Central configuration: paths, column groups, and run constants.

Every column-name string, path, and tunable constant used by the pipeline lives here so
the rest of the package never hard-codes schema knowledge.
"""

from pathlib import Path

# --- Paths (anchored to the repo root, independent of the working directory) ---
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "raw_data" / "Telco_customer_churn.xlsx"
PROCESSED_DIR = ROOT_DIR / "processed_data"
CLEAN_DATA_PATH = PROCESSED_DIR / "telco_churn_clean.csv"

# --- Run constants ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Target ---
TARGET = "Churn Value"

# --- Columns removed before modelling (see notebook/EDA.ipynb for the evidence) ---
CONSTANT_COLS = ["Count", "Country", "State"]
LEAKAGE_COLS = ["Churn Label", "Churn Score", "Churn Reason"]
IDENTIFIER_COLS = ["CustomerID", "Lat Long", "Zip Code", "City"]
VENDOR_SCORE_COLS = ["CLTV"]
DROP_COLS = CONSTANT_COLS + LEAKAGE_COLS + IDENTIFIER_COLS + VENDOR_SCORE_COLS

# --- Category values that are structurally determined by another column ---
# "No internet service" is fully implied by Internet Service == "No"; collapse to "No".
INTERNET_ADDON_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]
NO_INTERNET_VALUE = "No internet service"
NO_PHONE_VALUE = "No phone service"
COLLAPSED_VALUE = "No"

# --- Feature groups consumed by the preprocessor ---
BINARY_YESNO_COLS = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Paperless Billing",
    *INTERNET_ADDON_COLS,
]
NOMINAL_COLS = ["Gender", "Internet Service", "Payment Method"]
NUMERIC_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Latitude",
    "Longitude",
]
CONTRACT_COL = "Contract"
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]

# Column that arrives as text and must be coerced to float.
NUMERIC_TEXT_COL = "Total Charges"
