import pandas as pd
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding="utf-8")

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
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
dataset_path  = os.path.join(BASE_DIR, "dataset", "news_dataset.csv")
model_path    = os.path.join(BASE_DIR, "model",   "fake_news_model.pkl")
tfidf_path    = os.path.join(BASE_DIR, "model",   "tfidf_vectorizer.pkl")

print("\n" + "*" * 60)
print("*   FAKE NEWS DETECTION — MODEL TRAINING                 *")
print("*" * 60)

# ─────────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────────
section("Step 1: Loading dataset")

try:
    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"  [OK] Loaded news_dataset.csv — {len(df)} rows")
except Exception as e:
    print(f"  [ERROR] Could not load dataset: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# STEP 2: Separate X and y
# ─────────────────────────────────────────────
section("Step 2: Separating features and labels")

X = df["content"].astype(str)
y = df["label"]

print(f"  X (content) : {len(X)} rows")
print(f"  y (label)   : {len(y)} rows")
print(f"  Label distribution:")
print(f"    Fake (0) : {(y == 0).sum()}")
print(f"    Real (1) : {(y == 1).sum()}")

# ─────────────────────────────────────────────
# STEP 3: Split into train / test
# ─────────────────────────────────────────────
section("Step 3: Splitting into training and testing sets")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"  Training set size : {len(X_train)} rows  (80%)")
print(f"  Testing  set size : {len(X_test)}  rows  (20%)")

# ─────────────────────────────────────────────
# STEP 4: TF-IDF Vectorization
# ─────────────────────────────────────────────
section("Step 4: Applying TF-IDF Vectorization")

tfidf = TfidfVectorizer(stop_words="english", max_features=50000)

X_train_tfidf = tfidf.fit_transform(X_train)   # fit on train only
X_test_tfidf  = tfidf.transform(X_test)         # transform test

print(f"  Vocabulary size   : {len(tfidf.vocabulary_)} features")
print(f"  Training matrix   : {X_train_tfidf.shape}")
print(f"  Testing  matrix   : {X_test_tfidf.shape}")

# ─────────────────────────────────────────────
# STEP 5: Train Logistic Regression
# ─────────────────────────────────────────────
section("Step 5: Training Logistic Regression model")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

print("  [OK] Model trained successfully.")

# ─────────────────────────────────────────────
# STEP 6: Make predictions
# ─────────────────────────────────────────────
section("Step 6: Making predictions on test data")

y_pred = model.predict(X_test_tfidf)
print("  [OK] Predictions made on test set.")

# ─────────────────────────────────────────────
# STEP 7 & 8: Calculate and display metrics
# ─────────────────────────────────────────────
section("Step 7 & 8: Evaluation Metrics")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy   : {accuracy  * 100:.2f}%")
print(f"  Precision  : {precision * 100:.2f}%")
print(f"  Recall     : {recall    * 100:.2f}%")
print(f"  F1-Score   : {f1        * 100:.2f}%")

print("\n  Confusion Matrix:")
print("                   Predicted Fake   Predicted Real")
print(f"  Actual Fake   :       {cm[0][0]:<10}       {cm[0][1]}")
print(f"  Actual Real   :       {cm[1][0]:<10}       {cm[1][1]}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake (0)", "Real (1)"]))

# ─────────────────────────────────────────────
# STEP 9 & 10: Save model and vectorizer
# ─────────────────────────────────────────────
section("Step 9 & 10: Saving model and vectorizer")

joblib.dump(model, model_path)
print(f"  [OK] Model saved     : {model_path}")

joblib.dump(tfidf, tfidf_path)
print(f"  [OK] Vectorizer saved: {tfidf_path}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
section("Final Summary")

print(f"  Training dataset size   : {len(X_train)} rows")
print(f"  Testing  dataset size   : {len(X_test)} rows")
print(f"  Accuracy                : {accuracy  * 100:.2f}%")
print(f"  Precision               : {precision * 100:.2f}%")
print(f"  Recall                  : {recall    * 100:.2f}%")
print(f"  F1-Score                : {f1        * 100:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"    True Positives  (Real correctly predicted)  : {cm[1][1]}")
print(f"    True Negatives  (Fake correctly predicted)  : {cm[0][0]}")
print(f"    False Positives (Fake predicted as Real)    : {cm[0][1]}")
print(f"    False Negatives (Real predicted as Fake)    : {cm[1][0]}")
print(f"\n  Saved files:")
print(f"    model/fake_news_model.pkl    [OK]")
print(f"    model/tfidf_vectorizer.pkl   [OK]")

print("\n" + "=" * 60)
print("  Model training complete.")
print("=" * 60 + "\n")
