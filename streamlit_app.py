import streamlit as st
import joblib
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Load NLTK resources
@st.cache_resource
def load_nltk_resources():
    # Force download to ensure they are available in Streamlit Cloud
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

load_nltk_resources()

# Load Models
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'adaboost_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'tfidf_vectorizer.pkl')
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        # Fallback for pickle
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer

try:
    model, vectorizer = load_models()
except Exception as e:
    st.error(f"Critical Error: Could not load models. {e}")
    st.stop()

# Preprocessing Function (Same as Flask app)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
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

# UI Application
st.title("📧 Email Spam Detector")
st.markdown("Enter an email message below to check if it's **Spam** or **Safe**.")

input_text = st.text_area("Email Content", height=200, placeholder="Paste email text here...")

if st.button("Analyze Email", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess
            cleaned_text = preprocess_text(input_text)
            
            if len(cleaned_text.split()) < 2:
                 st.warning("Text is too short or mostly contains stopwords. Please try a longer message.")
            else:
                # Vectorize
                X_new = vectorizer.transform([cleaned_text])
                
                # Predict
                prediction = model.predict(X_new)[0]
                prob_spam = model.predict_proba(X_new)[0][1]
                
                # Results
                st.divider()
                if prediction == 1:
                    st.error(f"🚨 **SPAM DETECTED**")
                    st.progress(prob_spam, text=f"Confidence: {prob_spam:.1%}")
                else:
                    st.success(f"✅ **NOT SPAM (SAFE)**")
                    st.progress(1 - prob_spam, text=f"Confidence: {(1 - prob_spam):.1%}")
                
                with st.expander("See Processed Text"):
                    st.code(cleaned_text)

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Model: AdaBoost")
