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
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
fake_path  = os.path.join(BASE_DIR, "dataset", "Fake.csv")
true_path  = os.path.join(BASE_DIR, "dataset", "True.csv")
out_path   = os.path.join(BASE_DIR, "dataset", "news_dataset.csv")

print("\n" + "*" * 60)
print("*   FAKE NEWS DETECTION — DATASET PREPARATION            *")
print("*" * 60)

# ─────────────────────────────────────────────
# STEP 1: Load both CSV files
# ─────────────────────────────────────────────
section("Step 1: Loading datasets")

try:
    fake_df = pd.read_csv(fake_path, encoding="utf-8")
    print(f"  [OK] Loaded Fake.csv  — {len(fake_df)} rows")
except Exception as e:
    print(f"  [ERROR] Could not load Fake.csv: {e}")
    sys.exit(1)

try:
    true_df = pd.read_csv(true_path, encoding="utf-8")
    print(f"  [OK] Loaded True.csv  — {len(true_df)} rows")
except Exception as e:
    print(f"  [ERROR] Could not load True.csv: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# STEP 2: Add label column
#   Fake = 0  |  True = 1
# ─────────────────────────────────────────────
section("Step 2: Adding labels")

fake_df["label"] = 0   # 0 = Fake
true_df["label"] = 1   # 1 = Real

print("  Fake.csv  → label = 0 (Fake)")
print("  True.csv  → label = 1 (Real)")

# ─────────────────────────────────────────────
# STEP 3: Combine both datasets
# ─────────────────────────────────────────────
section("Step 3: Combining datasets")

df = pd.concat([fake_df, true_df], ignore_index=True)
print(f"  Combined total rows : {len(df)}")

# ─────────────────────────────────────────────
# STEP 4 & 5: Create 'content' column
#   content = title + " " + text
# ─────────────────────────────────────────────
section("Step 4 & 5: Creating 'content' column (title + text)")

df["content"] = df["title"].astype(str).str.strip() + " " + df["text"].astype(str).str.strip()
print("  'content' column created by combining 'title' and 'text'.")

# ─────────────────────────────────────────────
# STEP 6: Remove rows where content is empty or missing
# ─────────────────────────────────────────────
section("Step 6: Removing empty or missing content rows")

before = len(df)
df["content"] = df["content"].str.strip()
df = df[df["content"].notna()]
df = df[df["content"] != ""]
df = df[df["content"] != "nan nan"]
after = len(df)
print(f"  Rows before : {before}")
print(f"  Rows after  : {after}")
print(f"  Removed     : {before - after} row(s)")

# ─────────────────────────────────────────────
# STEP 7: Remove duplicate rows based on content
# ─────────────────────────────────────────────
section("Step 7: Removing duplicate rows")

before = len(df)
df = df.drop_duplicates(subset=["content"])
after = len(df)
print(f"  Rows before : {before}")
print(f"  Rows after  : {after}")
print(f"  Removed     : {before - after} duplicate(s)")

# ─────────────────────────────────────────────
# STEP 8: Keep only 'content' and 'label'
# ─────────────────────────────────────────────
section("Step 8: Keeping only 'content' and 'label' columns")

df = df[["content", "label"]]
print(f"  Columns kept: {list(df.columns)}")

# ─────────────────────────────────────────────
# STEP 9: Shuffle with fixed random state
# ─────────────────────────────────────────────
section("Step 9: Shuffling dataset (random_state=42)")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("  Dataset shuffled with random_state=42 (reproducible).")

# ─────────────────────────────────────────────
# STEP 10: Save to dataset/news_dataset.csv
# ─────────────────────────────────────────────
section("Step 10: Saving to dataset/news_dataset.csv")

df.to_csv(out_path, index=False, encoding="utf-8")
print(f"  [OK] Saved to: {out_path}")

# ─────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────
section("Final Report")

fake_count = (df["label"] == 0).sum()
real_count = (df["label"] == 1).sum()

print(f"  Total rows            : {len(df)}")
print(f"  Fake news articles    : {fake_count}  (label = 0)")
print(f"  Real news articles    : {real_count}  (label = 1)")
print(f"  Column names          : {list(df.columns)}")
print(f"\n  First 5 rows:")
print(df.head(5).to_string())

print("\n" + "=" * 60)
print("  Dataset preparation complete.")
print("  File saved: dataset/news_dataset.csv")
print("=" * 60 + "\n")
