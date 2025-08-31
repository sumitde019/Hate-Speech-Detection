# 🚨 Hate Speech & Offensive Language Detector

- A Streamlit web app that detects Hate Speech, Offensive Language, or Neutral text in tweets.  
- It uses **NLTK for preprocessing** and a **Decision Tree Classifier** from Scikit-Learn.  
- This project demonstrates how to clean text, train a classifier, and deploy an interactive ML app.

---

## 🔹 Demo
👉 [Live App on Streamlit](https://hate-speech-detection-019.streamlit.app/) 

---

## 🔹 Project Overview
This project is designed to **analyze text or tweets** and classify them into one of three categories:

- 🟥 **Hate Speech**  
- 🟧 **Offensive Language**  
- 🟩 **No Hate/Offensive**  

The model is trained using the **Twitter Hate Speech dataset**, and preprocessing includes:
- Lowercasing  
- Removing punctuation, numbers, links, and HTML tags  
- Stopword removal  
- Stemming with Snowball Stemmer  

The web app provides a simple text input box where users can enter any text, and the trained classifier predicts its category.

---

## 🔹 Project Structure
```
my-app/
  |── app.py # Streamlit app (main entry point)
  │── twitter.csv # Dataset used for training
  │── requirements.txt # Python dependencies
  │── README.md # Project documentation
  │── Hate_Speech_Detection_Model.ipynb
```

---

## 🔹 How It Works

1. **User Input**  
   - The user enters text in a Streamlit input box.

2. **Text Cleaning**  
   - Convert text to lowercase.  
   - Remove links, numbers, punctuation, HTML tags, and newlines.  
   - Remove stopwords using **NLTK**.  
   - Apply **stemming** using the Snowball Stemmer.

3. **Feature Extraction**  
   - Convert the cleaned text into numerical features using **CountVectorizer**.

4. **Prediction**  
   - Pass the features to the trained **Decision Tree Classifier**.

5. **Output**  
   - Display the prediction: **Hate Speech**, **Offensive Language**, or **No Hate/Offensive**.
