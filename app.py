from flask import Flask, request, render_template
import joblib
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings

# Suppress scikit-learn version mismatch warning (common when using pickled models)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Download required NLTK data if missing (only once)
nltk_resources = ['stopwords', 'punkt']
for resource in nltk_resources:
    try:
        path = f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}'
        nltk.data.find(path)
    except LookupError:
        print(f"Downloading NLTK data: {resource}")
        nltk.download(resource, quiet=True)

app = Flask(__name__)

# Model & vectorizer paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'adaboost_model.pkl')       # change to .joblib later
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# Load model and vectorizer once at startup
print("Loading model and vectorizer...")
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model & vectorizer loaded successfully using joblib")
except Exception:
    try:
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model & vectorizer loaded using pickle (fallback)")
    except Exception as e:
        print(f"CRITICAL ERROR – failed to load model/vectorizer:\n{e}")
        print("Solutions:")
        print("1. Make sure the .pkl files exist in the correct folder")
        print("2. Retrain and save using joblib.dump(model, 'adaboost_model.joblib')")
        exit(1)

print(f"Expected number of features: {len(vectorizer.vocabulary_)}")

# Global preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """Text preprocessing pipeline – must match exactly the training pipeline"""
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [stemmer.stem(w) for w in tokens]
    
    return ' '.join(tokens)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',
                          prediction=None,
                          confidence=None,
                          message='',
                          cleaned='',
                          error=None)


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '').strip()

    if not message:
        return render_template('index.html',
                              error="Please enter a message or email content first",
                              message=message)

    try:
        cleaned = preprocess_text(message)

        if len(cleaned.split()) < 3:
            return render_template('index.html',
                                  error="Text is too short or meaningless after preprocessing",
                                  message=message,
                                  cleaned=cleaned)

        X_new = vectorizer.transform([cleaned])

        pred = model.predict(X_new)[0]
        prob_spam = model.predict_proba(X_new)[0][1]   # probability of spam class

        # ── Full English result ───────────────────────────────────────────────
        result = "SPAM" if pred == 1 else "NOT SPAM"
        confidence_str = f"{prob_spam:.1%} chance of being SPAM" if pred == 1 else f"{(1 - prob_spam):.1%} chance of being SAFE"

        return render_template('index.html',
                              prediction=result,
                              confidence=confidence_str,
                              message=message,
                              cleaned=cleaned,
                              error=None)

    except Exception as e:
        return render_template('index.html',
                              error=f"Prediction error occurred: {str(e)}",
                              message=message)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   SPAM EMAIL DETECTION WEB APPLICATION")
    print("   Model: AdaBoost + TF-IDF")
    print("   Open →  http://127.0.0.1:5000")
    print("   (accessible from other devices on the same network)")
    print("="*70 + "\n")

    # Run on 0.0.0.0 so it can be accessed from phone / other laptop
    app.run(debug=True, host='0.0.0.0', port=5000)