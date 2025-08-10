
import os
import re
import joblib
import numpy as np
from flask import Flask, request, render_template_string
import spacy

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'artifacts', 'model.pkl')
VECT_PATH  = os.path.join(APP_DIR, 'artifacts', 'tfidf.pkl')

nlp = spacy.load("en_core_web_sm", disable=['ner','parser'])
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

def spacy_clean(text: str) -> str:
    text = text.lower()
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
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[0,1]
    else:
        p = model.predict_proba(X)[0,1]
    conf = abs(p - 0.5) * 2
    if conf < neutral_margin*2:
        label = 'Neutral'
    else:
        label = 'Positive' if p >= 0.5 else 'Negative'
    return label, float(p), float(conf)

HTML = '''
<!doctype html>
<title>Real-Time Sentiment</title>
<h2>Real-Time Sentiment (IMDB model)</h2>
<form method="post" action="/predict">
  <textarea name="text" rows="6" cols="80" placeholder="Type or paste text here..."></textarea><br><br>
  <button type="submit">Analyze</button>
</form>
{% if result %}
  <h3>Result</h3>
  <p><b>Label:</b> {{ result['label'] }}</p>
  <p><b>Positive Probability:</b> {{ '{:.3f}'.format(result['prob']) }}</p>
  <p><b>Confidence:</b> {{ '{:.3f}'.format(result['conf']) }}</p>
{% endif %}
'''

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text','')
    label, prob, conf = predict_with_confidence(text)
    return render_template_string(HTML, result={'label':label,'prob':prob,'conf':conf})

if __name__ == '__main__':
    # Run:  python app.py
    # Then open http://127.0.0.1:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
