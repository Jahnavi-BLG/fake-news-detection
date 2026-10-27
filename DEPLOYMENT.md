# 🚀 Deployment Guide — Fake News Detection Web Application

This document provides step-by-step instructions to deploy the **Fake News Detection** Flask application to production hosting platforms.

---

## 📋 Pre-Deployment Overview

The application is pre-configured with the following cloud-ready features:
1. **Dynamic Port Binding**: `app.py` reads `PORT` from the environment and binds to `0.0.0.0`.
2. **Production WSGI Server**: `gunicorn` is configured in `requirements.txt` and `Procfile`.
3. **Pre-trained Model Loading**: The app loads `model/fake_news_model.pkl` (400 KB) and `model/tfidf_vectorizer.pkl` (1.86 MB) directly into memory at startup. **It does not retrain models on server launch.**
4. **Health Check Route**: `/health` is available for uptime probes and health monitoring.
5. **Cross-Platform Paths**: All paths use `os.path.join()` relative to the project root directory.

---

## 📦 Files to Push to GitHub

### ✅ Required Files for Deployment
These files **must** be pushed to your repository:
```
Fake-News-Detection/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── Procfile                   # Cloud startup command
├── .gitignore                 # Excludes caches & virtualenvs
├── README.md                  # Project documentation
├── DEPLOYMENT.md              # Deployment guide
├── test_app.py                # Automated tests
├── model/
│   ├── fake_news_model.pkl    # Serialized ML model (~400 KB)
│   └── tfidf_vectorizer.pkl   # Serialized Vectorizer (~1.86 MB)
├── templates/                 # All HTML Jinja2 templates
│   ├── base.html
│   ├── index.html
│   ├── analyze.html
│   ├── how_it_works.html
│   ├── performance.html
│   └── about.html
└── static/
    └── style.css              # Custom styling
```

### ℹ️ Dataset Files (`dataset/*.csv`)
- **Not required for deployment**: The deployed web application only needs the pre-trained `.pkl` files in the `model/` folder.
- The raw CSV files (`Fake.csv`, `True.csv`, `news_dataset.csv` ~214 MB total) are only needed if you want to run `prepare_dataset.py` or `train_model.py` locally.
- *Tip*: Due to GitHub file size guidelines (files over 50MB trigger warnings and over 100MB fail to push), keeping dataset CSV files off GitHub or using Git LFS is recommended.

---

## 🌐 Option 1: Deploying to Render (Recommended — Free & Easy)

Render is one of the easiest and most reliable platforms to host Python Flask applications with automatic GitHub deployments.

### Step 1: Push Code to GitHub
1. Initialize git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Fake News Detection AI Project"
   ```
2. Create a new repository on [GitHub](https://github.com/new).
3. Link and push to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Create a Web Service on Render
1. Go to [dashboard.render.com](https://dashboard.render.com/) and sign in with GitHub.
2. Click **New +** → **Web Service**.
3. Select your GitHub repository (`Fake-News-Detection`) and click **Connect**.
4. Configure the deployment settings:
   - **Name**: `fake-news-detector` (or your preferred name)
   - **Region**: Select the region closest to you (e.g., Singapore, Frankfurt, Oregon)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank)*
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`
5. Optional (Health Check):
   - Expand **Advanced** → set **Health Check Path** to `/health`.
6. Click **Create Web Service**.

Render will now build your project and deploy your web app. Your live URL will look like:
`https://fake-news-detector.onrender.com`

---

## 🌐 Option 2: Deploying to Railway

1. Sign in to [railway.app](https://railway.app/) using your GitHub account.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your `Fake-News-Detection` repository.
4. Railway will automatically detect the `Procfile` and `requirements.txt`.
5. Under **Settings** → **Networking** → Click **Generate Domain**.
6. Your application will be live at `https://your-app-name.up.railway.app`.

---

## 🌐 Option 3: Deploying to Heroku

### Using Heroku CLI:
1. Install the Heroku CLI and login:
   ```bash
   heroku login
   ```
2. Create a Heroku application:
   ```bash
   heroku create fake-news-ai-app
   ```
3. Deploy your code:
   ```bash
   git push heroku main
   ```
4. Ensure at least one web dyno is running:
   ```bash
   heroku ps:scale web=1
   ```
5. Open your live application:
   ```bash
   heroku open
   ```

---

## 🌐 Option 4: Deploying to PythonAnywhere

1. Sign up for a free account at [pythonanywhere.com](https://www.pythonanywhere.com/).
2. Open a **Bash Console** and clone your repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Go to the **Web** tab:
   - Click **Add a new web app** → Select **Manual Configuration** → Select **Python 3.10** (or higher).
   - Set **Virtualenv path** to: `/home/YOUR_USERNAME/YOUR_REPO_NAME/venv`
   - Set **Source code path** to: `/home/YOUR_USERNAME/YOUR_REPO_NAME`
5. In the **WSGI configuration file** (click the link under the WSGI section):
   Replace the content with:
   ```python
   import sys
   import os

   path = '/home/YOUR_USERNAME/YOUR_REPO_NAME'
   if path not in sys.path:
       sys.path.append(path)

   from app import app as application
   ```
6. Click **Reload <your-username>.pythonanywhere.com** to launch.

---

## 🐳 Option 5: Container Deployment (Docker)

If you wish to deploy using Docker, create a `Dockerfile`:

```dockerfile
# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
```

### Build and Run:
```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

---

## 🔍 Verification & Health Check

After deployment, test that all endpoints are functioning properly:
- `GET /` — Home landing page
- `GET /analyze` — Interactive detection dashboard
- `POST /analyze` — Prediction request with article headline/text
- `GET /how-it-works` — System architecture and NLP workflow
- `GET /performance` — Evaluation metrics and Confusion Matrix
- `GET /about` — Project specifications & scope
- `GET /health` — Deployment health monitor (returns 200 OK)

---

## 🛠️ Common Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Application Error` / `H10 Crash`** | Host bound to `127.0.0.1` instead of `0.0.0.0` or missing `PORT` | Resolved in `app.py` and `Procfile`. Verify `gunicorn app:app` is configured. |
| **`FileNotFoundError: model/fake_news_model.pkl`** | Model files not pushed to GitHub | Make sure `model/fake_news_model.pkl` and `model/tfidf_vectorizer.pkl` are tracked by git and committed. |
| **`Memory Limit Exceeded / OOM`** | Model or dataset loading consumed too much RAM | The app only loads the pre-trained model (~2.3 MB total in RAM), well within the 512 MB free tier limit. |
| **`Static CSS Not Loading`** | Hardcoded asset paths | Verified: all static files use `{{ url_for('static', filename='style.css') }}`. |
