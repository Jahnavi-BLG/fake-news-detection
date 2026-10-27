import unittest
import os
import sys
import joblib

# Fix Windows terminal encoding for console output if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import app, model, tfidf

class TestFakeNewsDetectionApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_model_and_tfidf_loaded(self):
        """Verify model and TF-IDF vectorizer are loaded and intact."""
        self.assertIsNotNone(model)
        self.assertIsNotNone(tfidf)
        print("[OK] Model and TF-IDF vectorizer loaded successfully.")

    def test_health_endpoint(self):
        """Test GET request to health check route."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("running", content.lower())
        print("[OK] Health endpoint returns 200 OK.")

    def test_home_page_load(self):
        """Test GET request to home route."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("FAKE NEWS DETECTION", content)
        print("[OK] Home page GET request renders correctly.")

    def test_analyze_page_load(self):
        """Test GET request to analyze route."""
        response = self.app.get('/analyze')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("Article Verification Dashboard", content)
        print("[OK] Analyze page GET request renders correctly.")

    def test_empty_input_validation(self):
        """Test POST request with empty text on /analyze."""
        response = self.app.post('/analyze', data={'news_title': '', 'news_text': ''})
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("Please enter news title or article content before checking.", content)
        print("[OK] Empty input validation error works.")

    def test_real_news_prediction_with_confidence(self):
        """Test POST request with a real news example and confidence calculation."""
        real_title = "Global Climate Summit Concludes with Landmark Renewable Energy Agreement"
        real_text = "Delegates from over 190 nations reached a landmark agreement today at the international climate conference. The new framework commits member countries to accelerating clean energy transition goals, expanding solar and wind infrastructure investments, and phasing out coal subsidies over the next decade. Industry analysts described the policy accord as a significant structural shift in global environmental governance."
        
        response = self.app.post('/analyze', data={'news_title': real_title, 'news_text': real_text})
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("REAL NEWS", content)
        self.assertIn("Confidence", content)
        self.assertIn("Probability Breakdown", content)
        print("[OK] Real news prediction returned REAL NEWS with confidence & probability scores.")

    def test_fake_news_prediction(self):
        """Test POST request with a fake news example."""
        fake_title = "BREAKING: Scientists Discover Secret Island Inhabited By Living Dinosaurs"
        fake_text = "Unbelievable secret reports confirm that a covert team of researchers stumbled upon a hidden volcanic island in the South Pacific where prehistoric dinosaurs survived extinction. Government officials have reportedly classified all radar footage and restricted air space over the area to prevent the public from discovering the truth. Share this shock report before it gets taken down by mainstream media!"
        
        response = self.app.post('/analyze', data={'news_title': fake_title, 'news_text': fake_text})
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("FAKE NEWS", content)
        print("[OK] Fake news prediction returned FAKE NEWS.")

    def test_performance_page_load(self):
        """Test GET request to performance route."""
        response = self.app.get('/performance')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("Model Evaluation & Metrics", content)
        self.assertIn("98.73%", content)
        print("[OK] Performance page GET request renders correctly.")

    def test_about_page_load(self):
        """Test GET request to about route."""
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("About the Project", content)
        print("[OK] About page GET request renders correctly.")

    def test_how_it_works_page_load(self):
        """Test GET request to how-it-works route."""
        response = self.app.get('/how-it-works')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')
        self.assertIn("How It Works", content)
        print("[OK] How-It-Works page GET request renders correctly.")

    def test_api_predict_json(self):
        """Test POST request to /api/predict with JSON payload."""
        payload = {
            "title": "Scientists Discover Groundbreaking Clean Energy Breakthrough",
            "text": "Researchers at international universities have announced a scalable method for green hydrogen production."
        }
        response = self.app.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("prediction", data)
        self.assertIn("probabilities", data)
        self.assertIn("linguistics", data)
        print("[OK] REST API /api/predict returns valid JSON response.")

    def test_api_predict_empty_validation(self):
        """Test POST request to /api/predict with empty payload."""
        response = self.app.post('/api/predict', json={"title": "", "text": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        print("[OK] REST API /api/predict validation error handled correctly.")

if __name__ == "__main__":
    unittest.main()
