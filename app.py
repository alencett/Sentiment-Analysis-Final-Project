import os
import re
import io
import base64
import joblib
import numpy as np
from flask import Flask, request, render_template, url_for
import spacy
from wordcloud import WordCloud

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'artifacts', 'model.pkl')
VECT_PATH  = os.path.join(APP_DIR, 'artifacts', 'tfidf.pkl')

# --------- Load NLP + Model artifacts ----------
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# --------- Helpers ----------
def spacy_clean(text: str) -> str:
    text = text.lower()
    # remove HTML breaks
    text = re.sub(r'<br\s*/?>', ' ', text)
    # remove digits and non-letters
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    doc = nlp(text)
    lemmas = []
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.is_stop:
            continue
        lemma = tok.lemma_.strip()
        if lemma:
            lemmas.append(lemma)
    return ' '.join(lemmas)

def predict_with_confidence(text: str, neutral_margin=0.1):
    clean = spacy_clean(text)
    X = vectorizer.transform([clean])
    # Positive class probability
    p = model.predict_proba(X)[0, 1] if hasattr(model, "predict_proba") else 0.5
    # Confidence derived from distance to 0.5
    conf = abs(p - 0.5) * 2  # [0,1]
    # Neutral rule
    if conf < neutral_margin * 2:
        label = 'Neutral'
    else:
        label = 'Positive' if p >= 0.5 else 'Negative'
    # A ternary view for charting (sum not guaranteed to 1 but useful)
    pos_p = p
    neg_p = 1 - p
    neutral_p = max(0.0, 1 - conf)
    return label, float(p), float(conf), float(pos_p), float(neg_p), float(neutral_p), clean

def wordcloud_base64(text: str):
    if not text or not text.strip():
        return None
    wc = WordCloud(width=900, height=600, background_color="white").generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def top_local_keywords(clean_text: str, top_k=10):
    """
    Show key influencing tokens for THIS input by:
    - Vectorizing the cleaned text with the TF-IDF used in training.
    - Projecting onto the logistic regression coefficients (feature contributions).
    """
    if not hasattr(model, "coef_"):
        return [], []
    X = vectorizer.transform([clean_text])
    coefs = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    # contributions per feature in this input: x_i * w_i
    X_coo = X.tocoo()
    contribs = {}
    for i, j, v in zip(X_coo.row, X_coo.col, X_coo.data):
        contribs[feature_names[j]] = v * coefs[j]

    if not contribs:
        return [], []

    # Sort by contribution
    items = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    top_pos = [f"{k} ({v:.3f})" for k, v in items[:top_k]]
    top_neg = [f"{k} ({v:.3f})" for k, v in items[-top_k:][::-1]]
    return top_pos, top_neg

# --------- Flask app ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, text="")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text.", result=None, text="")

    label, prob, conf, pos_p, neg_p, neutral_p, clean = predict_with_confidence(text)

    # word cloud from cleaned text
    wc_img = wordcloud_base64(clean)

    # top local keywords (positive and negative contributors)
    top_pos, top_neg = top_local_keywords(clean)

    result = {
        "label": label,
        "prob": prob,              # positive probability
        "conf": conf,              # confidence [0..1]
        "chart": {                 # values used by Chart.js
            "positive": pos_p,
            "negative": neg_p,
            "neutral": neutral_p
        },
        "wordcloud": wc_img,
        "top_pos": top_pos,
        "top_neg": top_neg
    }

    return render_template("index.html", result=result, text=text)

@app.route("/clear", methods=["POST"])
def clear():
    return render_template("index.html", result=None, text="")

if __name__ == '__main__':
    # Run:  python app.py
    # Then open http://127.0.0.1:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
