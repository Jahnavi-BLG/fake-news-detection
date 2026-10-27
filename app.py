import os
import re
import math
import joblib
import socket
import ipaddress
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Initialize Flask app
app = Flask(__name__)

# Base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# Load trained model and TF-IDF vectorizer using joblib
model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024  # 3 MB is plenty for an article page


def normalize_url(raw):
    """Add https:// when the user pastes 'example.com/news' without a scheme."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "https://" + raw
    return raw


def is_safe_public_url(url):
    """Block localhost / private network targets (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        for info in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except socket.gaierror:
        return True  # let requests report the DNS error with a friendly message
    return True


def extract_article_from_html(html):
    """Return (title, body_text) from raw HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # --- Title: prefer og:title, then <h1>, then <title>
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "twitter:title"})
    if og and og.get("content"):
        title = og["content"].strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    title = re.sub(r"\s+", " ", title)

    # --- Remove non-content elements
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "nav",
                     "header", "footer", "aside", "button", "figure"]):
        tag.decompose()

    # --- Find the best container: <article>, else the element holding most <p> text
    container = soup.find("article")
    if container is None or len(container.get_text(strip=True)) < 300:
        best, best_len = None, 0
        for cand in soup.find_all(["main", "div", "section"]):
            plen = sum(len(p.get_text(strip=True)) for p in cand.find_all("p", recursive=False))
            if plen > best_len:
                best, best_len = cand, plen
        container = best or soup.body or soup

    paragraphs = []
    for p in container.find_all("p"):
        txt = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        if len(txt) >= 40:  # skip menu crumbs, captions, cookie one-liners
            paragraphs.append(txt)

    # Fallback: any paragraph on the page
    if len(" ".join(paragraphs)) < 200:
        paragraphs = [re.sub(r"\s+", " ", p.get_text(" ", strip=True))
                      for p in soup.find_all("p") if len(p.get_text(strip=True)) >= 40]

    # Last resort: meta description
    body = " ".join(paragraphs).strip()
    if len(body) < 100:
        meta = soup.find("meta", attrs={"property": "og:description"}) or soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            body = (body + " " + meta["content"].strip()).strip()

    return title, body


def compute_linguistic_metrics(text):
    """Analyze sensationalism, clickbait markers, and linguistic indicators in text."""
    if not text:
        return {
            "sensationalism_score": 0,
            "sensationalism_level": "Low",
            "caps_ratio": 0.0,
            "caps_word_count": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "trigger_words_found": []
        }

    words = re.findall(r'\b[A-Za-z]+\b', text)
    total_words = len(words)

    # Capitalized words (excluding single 'I' or 'A')
    all_caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_ratio = round((len(all_caps_words) / total_words) * 100, 1) if total_words > 0 else 0.0

    exclamation_count = text.count("!")
    question_count = text.count("?")

    # Common clickbait / sensationalist trigger patterns
    trigger_words = {
        "shocking", "unbelievable", "secret", "exposed", "conspiracy", "urgent",
        "miracle", "viral", "banned", "horrifying", "bombshell", "undeniable",
        "won't believe", "hidden truth", "they don't want you to know",
        "jaw-dropping", "proof", "insane", "destroy", "traitor", "scandal", "apocalypse"
    }

    text_lower = text.lower()
    found_triggers = [t for t in trigger_words if t in text_lower]

    # Calculate weighted sensationalism score (0 - 100)
    score = 0
    score += min(35, len(all_caps_words) * 7)
    score += min(25, exclamation_count * 5)
    score += min(15, question_count * 3)
    score += min(35, len(found_triggers) * 12)
    score = min(100, score)

    if score >= 60:
        level = "High"
    elif score >= 25:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "sensationalism_score": score,
        "sensationalism_level": level,
        "caps_ratio": caps_ratio,
        "caps_word_count": len(all_caps_words),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "trigger_words_found": found_triggers[:5]
    }


# Phrases typical of professional reporting (attribution, sourcing, concrete facts)
CREDIBILITY_CUES = [
    r"\bsaid\b", r"\bsays\b", r"\baccording to\b", r"\bannounced\b", r"\bstatement\b",
    r"\breported\b", r"\bofficials?\b", r"\bspokes(?:man|woman|person)\b", r"\bresearchers?\b",
    r"\bstudy\b", r"\bpublished\b", r"\bpeer[- ]reviewed\b", r"\bministry\b", r"\bgovernment\b",
    r"\bpercent\b", r"%", r"\bagreement\b", r"\banalysts?\b", r"\bcommittee\b", r"\bgovernor\b",
    r"\breporters?\b", r"\bcourt\b", r"\bpolice\b", r"\bcompany\b", r"\bmonday|tuesday|wednesday|thursday|friday|saturday|sunday\b",
]


def credibility_adjustment(text, linguistics, model_logit):
    """
    Returns (logit_adjustment, signals). Positive pushes toward REAL, negative toward FAKE.

    The trained model learned its 'real' signal mostly from wire-service (Reuters-style) writing,
    so ordinary real news that is written differently often lands near 50%. This layer adds bounded,
    explainable evidence from journalistic style so such articles are not left at ~50%.
    """
    signals = []
    adj = 0.0
    words = re.findall(r"\b[A-Za-z']+\b", text)
    n_words = len(words)
    text_lower = text.lower()

    cue_hits = sum(1 for pat in CREDIBILITY_CUES if re.search(pat, text_lower))
    if cue_hits:
        adj += min(2.0, 0.4 * cue_hits)
        signals.append(f"{cue_hits} professional-reporting cues (attribution, sources, officials, data)")

    numbers = len(re.findall(r"\b\d[\d,.]*\b", text))
    if numbers >= 2:
        adj += min(0.6, 0.2 * numbers)
        signals.append("contains concrete figures / dates")

    if re.search(r"[\"\u201c][^\"\u201d]{15,}[\"\u201d]", text):
        adj += 0.3
        signals.append("contains direct quotations")

    sens = linguistics.get("sensationalism_score", 0)
    if sens == 0 and n_words >= 25:
        adj += 0.5
        signals.append("neutral tone (no clickbait markers)")
    elif sens >= 25:
        adj -= min(3.0, sens / 20)
        signals.append(f"sensationalist language (risk {sens}%)")

    if linguistics.get("caps_ratio", 0) >= 8:
        adj -= 0.8
    if linguistics.get("exclamation_count", 0) >= 3:
        adj -= 0.8

    # Never let style cues flip a strongly fake model verdict
    if model_logit < -2.5 and adj > 0:
        adj *= 0.4

    return max(-3.5, min(2.8, adj)), signals


def predict_news(title, text):
    """Core AI prediction pipeline with TF-IDF transformation and Logistic Regression inference."""
    combined_content = f"{title} {text}".strip()
    if not combined_content:
        return None, "Please enter news title or article content before checking."

    # Transform text input using saved TF-IDF vectorizer
    text_vectorized = tfidf.transform([combined_content])

    # Raw model score (log-odds of REAL)
    model_logit = float(model.decision_function(text_vectorized)[0])
    model_real_prob = 1.0 / (1.0 + math.exp(-model_logit))

    linguistics = compute_linguistic_metrics(combined_content)
    adj, signals = credibility_adjustment(combined_content, linguistics, model_logit)

    final_logit = model_logit + adj
    real_p = 1.0 / (1.0 + math.exp(-final_logit))
    prediction = 1 if real_p >= 0.5 else 0
    real_prob = round(real_p * 100, 2)
    fake_prob = round((1.0 - real_p) * 100, 2)
    confidence = round(max(fake_prob, real_prob), 2)
    result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    # Compute article text analytics
    words = combined_content.split()
    word_count = len(words)
    char_count = len(combined_content)
    unique_words = len(set(w.lower() for w in words))
    reading_time = max(1, math.ceil(word_count / 200)) if word_count > 0 else 0

    stats = {
        "word_count": word_count,
        "char_count": char_count,
        "unique_words": unique_words,
        "reading_time": reading_time
    }

    # Extract top influential TF-IDF terms
    top_keywords = []
    try:
        feature_names = tfidf.get_feature_names_out()
        dense_vector = text_vectorized.toarray()[0]
        non_zero_indices = dense_vector.nonzero()[0]
        coefficients = model.coef_[0]

        word_scores = []
        for idx in non_zero_indices:
            word = feature_names[idx]
            weight = dense_vector[idx]
            coef = coefficients[idx]
            word_scores.append({
                "word": word,
                "weight": round(float(weight), 4),
                "lean": "REAL" if coef > 0 else "FAKE"
            })
        word_scores.sort(key=lambda x: x["weight"], reverse=True)
        top_keywords = word_scores[:6]
    except Exception:
        top_keywords = []

    return {
        "result": result,
        "prediction_class": prediction,
        "confidence": confidence,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "stats": stats,
        "top_keywords": top_keywords,
        "linguistics": linguistics,
        "signals": signals,
        "model_real_prob": round(model_real_prob * 100, 2)
    }, None

@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
def home():
    """Home landing page with project overview and feature highlights."""
    return render_template("index.html", active_page="home")

@app.route("/analyze", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def analyze():
    """Interactive article analysis page with AI prediction pipeline, confidence scores, and NLP feature analytics."""
    result = None
    news_title = ""
    news_text = ""
    error = None
    confidence = None
    real_prob = None
    fake_prob = None
    stats = None
    top_keywords = []
    linguistics = None
    signals = []

    if request.method == "POST":
        news_title = request.form.get("news_title", "").strip()
        news_text = request.form.get("news_text", "").strip()

        prediction_data, err = predict_news(news_title, news_text)
        if err:
            error = err
        else:
            result = prediction_data["result"]
            confidence = prediction_data["confidence"]
            real_prob = prediction_data["real_prob"]
            fake_prob = prediction_data["fake_prob"]
            stats = prediction_data["stats"]
            top_keywords = prediction_data["top_keywords"]
            linguistics = prediction_data["linguistics"]
            signals = prediction_data["signals"]

    return render_template(
        "analyze.html",
        active_page="analyze",
        result=result,
        news_title=news_title,
        news_text=news_text,
        error=error,
        confidence=confidence,
        real_prob=real_prob,
        fake_prob=fake_prob,
        stats=stats,
        top_keywords=top_keywords,
        linguistics=linguistics,
        signals=signals
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Public REST API endpoint for Fake News Prediction (JSON or form data)."""
    if request.is_json:
        data = request.get_json() or {}
        title = data.get("title", data.get("news_title", "")).strip()
        text = data.get("text", data.get("news_text", "")).strip()
    else:
        title = request.form.get("news_title", request.form.get("title", "")).strip()
        text = request.form.get("news_text", request.form.get("text", "")).strip()

    prediction_data, err = predict_news(title, text)
    if err:
        return jsonify({"success": False, "error": err}), 400

    return jsonify({
        "success": True,
        "title": title,
        "prediction": prediction_data["result"],
        "confidence": prediction_data["confidence"],
        "probabilities": {
            "real": prediction_data["real_prob"],
            "fake": prediction_data["fake_prob"]
        },
        "stats": prediction_data["stats"],
        "linguistics": prediction_data["linguistics"],
        "top_features": prediction_data["top_keywords"],
        "credibility_signals": prediction_data["signals"]
    }), 200

@app.route("/api/fetch-url", methods=["POST"])
def api_fetch_url():
    """Extracts headline and article text from a provided web URL."""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        raw_url = data.get("url", "")
    else:
        raw_url = request.form.get("url", "")

    url = normalize_url(raw_url)
    if not url or "." not in urlparse(url).netloc:
        return jsonify({"success": False, "error": "Please paste a valid article link, e.g. https://www.bbc.com/news/..."}), 400
    if not is_safe_public_url(url):
        return jsonify({"success": False, "error": "That address is not allowed. Please use a public news website link."}), 400

    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12, stream=True, allow_redirects=True)
        if resp.status_code in (401, 403, 429):
            return jsonify({"success": False, "error": f"The website blocked automatic access (HTTP {resp.status_code}). Open the article, copy its text and paste it below instead."}), 400
        if resp.status_code >= 400:
            return jsonify({"success": False, "error": f"The website returned an error (HTTP {resp.status_code}). Check the link and try again."}), 400

        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype.lower() and "text" not in ctype.lower():
            return jsonify({"success": False, "error": "This link is not a web page (it looks like a file or media). Please use an article link."}), 400

        content = b""
        for chunk in resp.iter_content(65536):
            content += chunk
            if len(content) > MAX_DOWNLOAD_BYTES:
                break
        encoding = resp.encoding if resp.encoding and "charset" in ctype.lower() else (resp.apparent_encoding or "utf-8")
        html = content.decode(encoding, errors="ignore")

        title, body_text = extract_article_from_html(html)

        if len(body_text) < 80:
            return jsonify({"success": False, "error": "Could not find readable article text on this page (it may need JavaScript or a login). Please paste the article text manually."}), 400

        return jsonify({"success": True, "title": title, "text": body_text[:5000], "source": urlparse(resp.url).netloc}), 200

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "The website took too long to respond. Please try again or paste the text manually."}), 400
    except requests.exceptions.SSLError:
        return jsonify({"success": False, "error": "Secure connection (SSL) to this website failed. Try another link or paste the text manually."}), 400
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": "Could not connect to that website. Check the link and your internet connection."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to fetch content from URL ({e})"}), 400

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "Fake News Detection AI API",
        "model_loaded": model is not None,
        "vectorizer_loaded": tfidf is not None
    }) if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html else "Fake News Detection App is running!"

@app.route("/how-it-works")
def how_it_works():
    """Detailed visual NLP and Machine Learning workflow page."""
    return render_template("how_it_works.html", active_page="how_it_works")

@app.route("/performance")
def performance():
    """Model performance and test set evaluation metrics page."""
    return render_template("performance.html", active_page="performance")

@app.route("/about")
def about():
    """Project overview, objectives, limitations, future scope, and tech stack page."""
    return render_template("about.html", active_page="about")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
