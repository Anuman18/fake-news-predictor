import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

nltk.download('stopwords')
stop_words = stopwords.words('english')

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.set_page_config(page_title="📰 Fake News Detector")
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline to check if it's **FAKE** or **REAL**.")

input_news = st.text_area("Enter News Text Here", height=200)

if st.button("Check"):
    if input_news.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(input_news)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        
        if prediction == "FAKE":
            st.error("🚨 This news appears to be **FAKE**.")
        else:
            st.success("✅ This news appears to be **REAL**.")
