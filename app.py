import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already present
nltk.download("stopwords")
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words("english"))

# ----------------------
# Data Loading & Preprocessing
# ----------------------
@st.cache_resource
def load_model():
    data = pd.read_csv("twitter.csv")

    # Map labels
    data["labels"] = data["class"].map({
        0: "Hate Speech",
        1: "Offensive Language",
        2: "No Hate and Offensive"
    })
    data = data[["tweet", "labels"]]

    # Clean text function
    def clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text

    data["tweet"] = data["tweet"].apply(clean)

    x = np.array(data["tweet"])
    y = np.array(data["labels"])

    cv = CountVectorizer()
    X = cv.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    return clf, cv

clf, cv = load_model()

# ----------------------
# Streamlit App
# ----------------------
st.title("🚨 Hate Speech & Offensive Language Detector")
st.write("This app classifies tweets/text into **Hate Speech, Offensive Language, or No Hate/Offensive**.")

# User Input
user_input = st.text_area("Enter text to analyze:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        data = cv.transform([user_input]).toarray()
        prediction = clf.predict(data)[0]
        st.success(f"### 📝 Prediction: **{prediction}**")
