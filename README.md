# 🚨 Hate Speech & Offensive Language Detector

> A Streamlit web app that detects Hate Speech, Offensive Language, or Neutral text in tweets.  
> It uses **NLTK for preprocessing** and a **Decision Tree Classifier** from Scikit-Learn.  
> This project demonstrates how to clean text, train a classifier, and deploy an interactive ML app.

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
