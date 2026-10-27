# ⚡ AI Fake News Detection System

An Artificial Intelligence web application that analyzes news headlines and article text to predict whether the content is **REAL** or **FAKE**. It combines a supervised machine learning classifier (TF-IDF + Logistic Regression) with a rule-based sensationalism heuristic, designed as an intelligent agent with a Flask web interface.

Project Based Learning (PBL) project for the **Artificial Intelligence Lab (20CS31L03)**, Geethanjali College of Engineering and Technology.

**Team:** B. Pravalika (24R11A05M1), B. L. G. Jahnavi (24R11A05M2), G. Sanjana (24R11A05M7)
**Guide:** A. Deepika, Assistant Professor, Department of CSE

---

## 🌟 Key Features

- **Interactive Verification Dashboard**: Real-time news authenticity analysis with confidence scores and class probability breakdown.
- **Rule-Based Sensationalism Score**: A hand-written heuristic (0-100, Low / Moderate / High) based on ALL-CAPS words, exclamation marks, question marks and trigger words.
- **NLP Text Analytics**: Live word counter, character counter, unique vocabulary metrics, and estimated reading time.
- **Explainable Predictions**: Highlights the top influential TF-IDF terms driving the model's prediction.
- **Model Performance & Evaluation Page**: Benchmark metrics (98.73% Accuracy, 98.43% Precision, 99.25% Recall, 98.84% F1-Score) and Confusion Matrix visualization with Chart.js.
- **JSON API**: `/api/predict` for programmatic access and `/api/fetch-url` to analyze a web page by its URL.
- **Cloud Ready**: Configured for deployment on Render, Railway, Heroku or any WSGI-compatible host.

---

## 🧠 AI Concepts Used

- **Intelligent agent**: PEAS description, learning agent (training, prediction, evaluation) and simple reflex agent (sensationalism rules)
- **Heuristic evaluation function**: weighted sum of text features for the sensationalism score
- **Rule-based reasoning**: IF-THEN thresholds (expressible as Horn clauses)
- **Reasoning under uncertainty**: class probabilities and confidence instead of a hard yes/no
- **Supervised learning**: Logistic Regression on TF-IDF features, with train/test evaluation (accuracy, precision, recall, F1, confusion matrix, ROC AUC)

---

## 📁 Project Structure

```
Fake-News-Detection/
├── app.py                     # Flask app, prediction and sensationalism heuristic
├── requirements.txt           # Python dependencies
├── Procfile                   # Gunicorn startup configuration for cloud hosts
├── .gitignore                 # Files excluded from Git
├── DEPLOYMENT.md              # Step-by-step deployment manual
├── test_app.py                # Automated unit tests
├── train_model.py             # Trains and evaluates the model and vectorizer
├── prepare_dataset.py         # Merges, cleans and prepares the dataset
├── inspect_dataset.py         # Inspects raw dataset properties
├── dataset/                   # Fake.csv, True.csv (NOT in Git, see Dataset section)
├── model/
│   ├── fake_news_model.pkl    # Trained Logistic Regression model (~400 KB)
│   └── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer (~1.86 MB)
├── templates/
│   ├── base.html              # Base layout with navigation
│   ├── index.html             # Landing page
│   ├── analyze.html           # News analysis dashboard
│   ├── how_it_works.html      # AI pipeline walkthrough
│   ├── performance.html       # Metrics, charts and confusion matrix
│   └── about.html             # Objectives, limitations and tech stack
└── static/
    └── style.css              # Responsive stylesheet and dark theme
```

---

## 🚀 Quick Start (Local Setup)

The trained model is already included, so you can run the app without the dataset.

### 1. Open the project folder
```bash
cd Fake-News-Detection
```

### 2. Create and activate a virtual environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```
Open your browser at `http://localhost:5000`

### 5. Run the tests
```bash
python test_app.py
```

---

## 📦 Dataset and Retraining (optional)

The dataset files are large (about 210 MB), so they are **not included in this repository** (`dataset/*.csv` is in `.gitignore`). You only need them if you want to retrain the model.

1. Download the Kaggle **"Fake and Real News"** dataset (`Fake.csv` and `True.csv`).
2. Place both files inside the `dataset/` folder.
3. Build the combined dataset and retrain:
   ```bash
   python prepare_dataset.py
   python train_model.py
   ```
This regenerates `model/fake_news_model.pkl` and `model/tfidf_vectorizer.pkl`.

> Tip: use the scikit-learn version in `requirements.txt` when retraining so the `.pkl` files load without version warnings.

---

## ☁️ Deployment Guide

- **Procfile**: `web: gunicorn app:app`
- **Dynamic Port**: Binds to `0.0.0.0` using the host's `$PORT` environment variable.
- **Health Check Endpoint**: `/health` for uptime and container probes.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Render, Railway, Heroku, PythonAnywhere and Docker.

---

## 📊 Model Evaluation Summary

Evaluated on a held-out 20% test set (7,821 articles).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.73%** |
| **Precision** | **98.43%** |
| **Recall** | **99.25%** |
| **F1-Score** | **98.84%** |
| **ROC AUC** | 0.9989 |
| **Vocabulary Features** | 50,000 TF-IDF features |
| **Prediction time** | ~35 ms (local machine) |

---

## ⚠️ Limitations

- The model learns word and style patterns from one dataset (mainly US political and world news). It does **not** verify facts, so a prediction is not proof that an article is true or false.
- Words such as "reuters" and weekday names are strong signals in the dataset, so accuracy may drop on articles from other sources or topics.
- English only. The app always returns a label, even for meaningless or non-English text.
- The sensationalism score uses hand-set weights that were not tuned on labelled data. It measures writing style, not truth.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.9+, Flask, Gunicorn
- **Machine Learning & NLP**: Scikit-learn, TF-IDF Vectorizer, Logistic Regression, Joblib, NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Chart.js
- **Testing**: Python `unittest`