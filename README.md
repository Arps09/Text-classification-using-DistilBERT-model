# 🌈 Sentivibe – AI Sentiment Analyzer

**Sentivibe** is a real-time, AI-powered web application that analyzes the **emotional sentiment** of any text input. Whether it’s a review, comment, or personal message — Sentivibe lets you *feel the vibe* of your words using cutting-edge NLP models.

---

## 🔍 Table of Contents

- [📌 Overview](#-overview)
- [🎯 Objective](#-objective)
- [🧠 How It Works](#-how-it-works)
- [🚀 Setup Instructions](#-setup-instructions)
- [📁 Project Structure](#-project-structure)
- [🛠️ Technologies Used](#️-technologies-used)
- [💡 Future Improvements](#-future-improvements)
- [🙌 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)

---

## 📌 Overview

Sentivibe is an interactive web app built using **Streamlit** and a fine-tuned **DistilBERT** model from Hugging Face. It provides real-time sentiment analysis of text, classifying it as **positive** or **negative** with helpful visual feedback.

---

## 🎯 Objective

The goal of this project is to:
- Enable easy and intuitive sentiment analysis using machine learning.
- Help users understand the emotional tone of their words.
- Explore the power of transformers for real-world NLP tasks.

---

## 🧠 How It Works

### 1. **Model Training**
- Trained a binary classifier using **DistilBERT** on labeled sentiment data (positive & negative).
- The model is fine-tuned using Hugging Face's `transformers` and PyTorch libraries in the `train.ipynb` notebook.

### 2. **Model Export**
- The trained model and tokenizer are saved in the `saved_model/` directory for use in the app.

### 3. **Web Application (Streamlit)**
- Built with Streamlit for a fast, interactive UI.
- Loads the model, takes user input, and predicts the sentiment.

### 4. **Prediction Logic**
- Tokenizes the input using the saved tokenizer.
- Runs inference on the DistilBERT model.
- Classifies sentiment as:
  - **1 → Positive 😊**
  - **0 → Negative 😞**
- Displays feedback with emojis, colors, and animations.

---

## 🚀 Setup Instructions

Follow these steps to run Sentivibe locally:

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/sentivibe.git
cd sentivibe
