import pandas as pd
import os
import sys

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# Helper: print a section header
# ─────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ─────────────────────────────────────────────
# Helper: inspect a single dataframe
# ─────────────────────────────────────────────
def inspect(name, filepath):
    section(f"Inspecting: {name}")

    # 1. Load the CSV
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        print(f"[OK] Loaded '{name}' successfully using UTF-8 encoding.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding="latin-1")
            print(f"[WARNING] UTF-8 failed. Loaded '{name}' using latin-1 encoding.")
            print("          This may indicate encoding issues in the file.")
        except Exception as e:
            print(f"[ERROR] Could not load '{name}': {e}")
            return
    except Exception as e:
        print(f"[ERROR] Could not load '{name}': {e}")
        return

    # 2. Shape (rows and columns)
    print(f"\n--- Shape (Rows x Columns) ---")
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")

    # 3. Column names
    print(f"\n--- Column Names ---")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    # 4. First 3 rows
    print(f"\n--- First 3 Rows ---")
    print(df.head(3).to_string())

    # 5. Missing values
    print(f"\n--- Missing Values Per Column ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found. [OK]")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"  [!] '{col}' has {count} missing value(s)")

    # 6. Duplicate rows
    print(f"\n--- Duplicate Rows ---")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        print("  No duplicate rows found. [OK]")
    else:
        print(f"  [!] {dup_count} duplicate row(s) found.")

    # 7. Check 'text' column
    print(f"\n--- Column Check: 'text' ---")
    if "text" in df.columns:
        print("  'text' column EXISTS. [OK]")
    else:
        print("  [!] 'text' column is MISSING.")

    # 8. Check 'title' column
    print(f"\n--- Column Check: 'title' ---")
    if "title" in df.columns:
        print("  'title' column EXISTS. [OK]")
    else:
        print("  [!] 'title' column is MISSING.")

    # 9. Data / encoding problems
    print(f"\n--- Data & Encoding Check ---")
    issues_found = False

    # Check for completely empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        print(f"  [!] Completely empty columns: {empty_cols}")
        issues_found = True

    # Check for columns with all identical values
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"  [!] Columns with only one unique value: {constant_cols}")
        issues_found = True

    # Check for odd characters in text/title if they exist
    for col in ["title", "text"]:
        if col in df.columns:
            sample = df[col].dropna().astype(str)
            garbled = sample[sample.str.contains(r'[^\x00-\x7F]', regex=True)]
            if not garbled.empty:
                print(f"  [!] '{col}' contains {len(garbled)} row(s) with non-ASCII characters.")
                print(f"       Sample: {garbled.iloc[0][:80]}")
                issues_found = True

    if not issues_found:
        print("  No obvious data or encoding problems found. [OK]")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("*   FAKE NEWS DETECTION — DATASET INSPECTION             *")
    print("*" * 60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(BASE_DIR, "dataset", "Fake.csv")
    true_path = os.path.join(BASE_DIR, "dataset", "True.csv")

    results = {}

    for name, path in [("Fake.csv", fake_path), ("True.csv", true_path)]:
        inspect(name, path)
        try:
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
            results[name] = {
                "rows": df.shape[0],
                "cols": df.shape[1],
                "missing": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated().sum()),
                "has_title": "title" in df.columns,
                "has_text":  "text"  in df.columns,
                "columns": list(df.columns),
            }
        except Exception:
            results[name] = None

    # ── SUMMARY ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  {'Property':<25} {'Fake.csv':<18} {'True.csv'}")
    print(f"  {'-'*25} {'-'*18} {'-'*15}")
    for key, label in [
        ("rows",       "Total Rows"),
        ("cols",       "Total Columns"),
        ("missing",    "Missing Values"),
        ("duplicates", "Duplicate Rows"),
        ("has_title",  "'title' Column Exists"),
        ("has_text",   "'text' Column Exists"),
    ]:
        fake_val = str(results["Fake.csv"][key]) if results.get("Fake.csv") else "N/A"
        true_val = str(results["True.csv"][key]) if results.get("True.csv") else "N/A"
        print(f"  {label:<25} {fake_val:<18} {true_val}")

    print("\n  Column names in Fake.csv :", results["Fake.csv"]["columns"] if results.get("Fake.csv") else "N/A")
    print("  Column names in True.csv :", results["True.csv"]["columns"] if results.get("True.csv") else "N/A")
    print("\n" + "=" * 60)
    print("  Inspection Complete.")
    print("=" * 60 + "\n")
