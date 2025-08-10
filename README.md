# Real‑Time Sentiment Analyzer — Final Project

### Presentation

Course: 2025S‑T3 BAM 3034 — Sentiment Analysis and Text Mining 01
Professor: Ishant Gupta
Institution: Lambton College, Mississauga
Date: August 2025

Group Members:

Alvaro Blanquicett — C0927639

Gina Ferrera — C0933111

Luis Carlos Muñoz — C0932513

Marian Velasquez — C0937278

### Introduction

This project delivers an end‑to‑end sentiment analysis system: a trained model on the IMDB Reviews dataset and a production‑ready Flask web app that provides real‑time sentiment predictions (Positive / Negative / Neutral), a confidence score, a probability chart, a word cloud, and key influencing words for explainability.

It is designed to be:

Reproducible (notebook for training, saved artifacts)

Deployable (local, Render, or Heroku)

Explainable (top TF‑IDF feature contributions per input)

User‑friendly (styled UI with About section, Clear/Analyze actions)

### Overview

What we built (step by step)
Data

Source: IMDB Reviews dataset (CSV) https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews 

Columns used: review, sentiment (positive/negative).

Preprocessing (spaCy)

Lowercasing

Remove HTML breaks (e.g., <br>, <br/>)

Remove digits and punctuation

Tokenization & stopword removal

Lemmatization

Output is a cleaned string of lemmas (column: clean).

Vectorization

TF‑IDF on cleaned text

N‑grams: (1,2), min_df=5, max_df=0.95

Modeling

Logistic Regression (primary model; probabilistic, fast, strong baseline)

Optional: LinearSVC with calibration (produces probabilities)

Optional (commented): BERT embeddings + linear model

Evaluation

Train/test split (stratified)

Accuracy, F1, classification report, confusion matrix

5‑fold stratified cross‑validation on F1

Explainability

Local contributions: for a given input, multiply TF‑IDF features by LR coefficients ⇒ list top positive/negative signals.

Word Cloud: built from the cleaned text of the input (or class corpora in the notebook).

Artifacts

Trained model → artifacts/model.pkl

Trained vectorizer → artifacts/tfidf.pkl

These are loaded at app startup for instant inference.

Backend (Flask)

Routes:

GET / — Home page (form + About)

POST /analyze — Run inference, render results

POST /clear — Clear the input and results

Logic:

Preprocess with spaCy → Vectorize → Predict probability → Compute confidence (distance from 0.5 → [0,1])

Label Neutral if confidence below a small margin (configurable)

Build word cloud & key features list

Pass a small JSON block with chart data to Chart.js (doughnut)

Frontend (HTML/CSS + Chart.js)

Pages (single page with results below the form):

Home/Results: Textarea, Analyze and Clear buttons, results panel (label, probability, confidence), doughnut chart, keywords, word cloud, and About section.

Design:

Dark theme, Inter font, subtle elevation, high contrast, responsive grid

Buttons with distinct styles (primary, neutral)

Accessible colors: green (positive), red (negative), amber (neutral)

### Technologies & Methods

Language: Python 3.11

Libraries:

spaCy (tokenization, lemmatization)

scikit‑learn (TF‑IDF, Logistic Regression, SVM, metrics)

WordCloud (word cloud image)

Flask (web server & templating)

Chart.js (probability doughnut chart on the client)

Model: Logistic Regression (probabilities used for confidence & neutral rule)

Vectorizer: TF‑IDF (unigrams + bigrams)

### Training Outcomes (example)

(Actual numbers will depend on the dataset version and RNG seeds; below are typical ranges.)

Logistic Regression

Accuracy: ~0.88–0.91

F1 (positive): ~0.88–0.91

5‑fold CV (F1): mean ~0.89 ± 0.01

SVM (calibrated)

Often comparable to LR; slightly different precision/recall balance.

BERT (optional)

Sketched as embeddings + linear classifier (commented). Fine‑tuning would improve accuracy but increases complexity and cost.

### Local Development

1) Prerequisites
Python 3.11 (recommended; wheels available for spaCy deps)

Git

2) Clone and set up
bash
git clone https://github.com/alencett/Sentiment-Analysis-Final-Project.git
cd Sentiment-Analysis-Final-Project

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
Note on spaCy model: requirements.txt installs the English model via the official wheel URL. No extra download step needed.

3) (Optional) Retrain the model
Open the notebook in notebooks/ (or the one you generated earlier), run through training, and overwrite:

bash
artifacts/model.pkl
artifacts/tfidf.pkl

4) Run locally
Option A: Flask dev server

bash
export FLASK_APP=app.py
flask run
# http://127.0.0.1:5000
Option B: Gunicorn (closer to production)

bash
gunicorn app:app --preload --workers 2 --threads 4 --timeout 120
# open http://127.0.0.1:8000

### Deploying to Render (recommended)
One‑time setup
Pin Python (already in repo):
.python-version
3.11.9

Ensure artifacts are committed:

bash
artifacts/model.pkl
artifacts/tfidf.pkl
requirements.txt contains the spaCy model wheel URL (so builds don’t need downloads):

ruby
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
Create the service
Go to Render → New → Web Service and connect the GitHub repo.

Environment: Python

Build Command:
nginx
pip install -r requirements.txt

Start Command:
bash
gunicorn app:app --preload --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
(Optional) Set env var PYTHON_VERSION=3.11.9 (redundant if .python-version is present).

Deploy.

Free plan note: After ~15 minutes of inactivity, the service sleeps. The first request will show “APPLICATION LOADING” while it wakes up (cold start). Upgrade the plan or ping the app periodically to keep it warm.

Deploying to Heroku
Files you need (already included)
Procfile

less
web: gunicorn app:app --preload --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
runtime.txt (optional but explicit)

python-3.11.9
Steps
bash
heroku login
heroku create your-sentiment-app --stack heroku-22
git push heroku main
heroku open
Or connect GitHub from the Heroku dashboard and enable auto‑deploys.

API & App Behavior
Endpoints
GET / — Render form

POST /analyze — Run prediction and render results

POST /clear — Clear text/result and render

Prediction logic
p = P(positive) from the model

Confidence = |p − 0.5| * 2 → [0,1]

If confidence < 2 * neutral_margin (default 0.1 → threshold 0.2) ⇒ Neutral

Otherwise: p ≥ 0.5 ⇒ Positive, else Negative

Explainability
Local feature contributions = (TF‑IDF value) × (LR coefficient) per term

Show top positive and negative contributors for the current input

### Security & Privacy

This is an academic project; no authentication or rate limiting is included by default.

Do not use the free public deployment for sensitive text.

If exposing publicly, consider adding:

Request size limits

Rate limiting

Logging/monitoring

HTTPS (Render/Heroku provide TLS)

### Known Limitations & Future Work

Domain generalization: Trained on movie reviews; predictions may be less accurate on unfamiliar domains (finance, medical, etc.).

Neutral threshold: Simple rule based on probability distance; could be replaced by a calibrated ternary classifier.

Explainability scope: Linear coefficients on TF‑IDF are intuitive but don’t capture complex semantics; upgrading to transformer‑based methods + SHAP/LIME would help.

Optional BERT fine‑tuning: Adds training complexity and compute requirements; left as future work.

### Acknowledgements

IMDB Reviews dataset for academic use.

spaCy, scikit‑learn, Flask, Chart.js, WordCloud communities.

Lambton College — Sentiment Analysis and Text Mining course.

### License
This repository is for educational purposes as part of a course final project.
If you plan to reuse or extend it, please check dataset licensing and credit the original authors/tools.