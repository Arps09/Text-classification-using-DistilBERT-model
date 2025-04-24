![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Project Status](https://img.shields.io/badge/Project%20Status-Completed-success)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

# 🧠 Text Classification using DistilBERT

## 📌 Overview  
This project focuses on **text classification** using the **DistilBERT** model, a lightweight and efficient version of BERT developed by Hugging Face. The notebook walks through the entire pipeline of natural language processing — from loading and preprocessing a text dataset to fine-tuning the transformer model and evaluating its performance. The aim is to build a robust, high-performing classifier that can generalize well to unseen text data using minimal resources and training time.

---

🚀 Live Demo
Experience the power of DistilBERT live! Click the link below to test the sentiment analysis tool in real-time using Streamlit:

🔗 [Try Sentivibe 💬](https://sentivibe.streamlit.app/)

---
🎥 Project Walkthrough
Check out the full explanation and walkthrough of the app, including development and usage:

▶️ [Watch on YouTube](https://youtu.be/DlBC7CpucT8?si=FX40yK66UWbcFZme)

## 🎯 Objectives  
- Load and prepare text data for classification tasks  
- Tokenize and encode text using **DistilBERT tokenizer**  
- Fine-tune the **`distilbert-base-uncased`** model on the dataset  
- Evaluate model performance using key NLP metrics  
- Predict outcomes on new, custom text samples  

---

## 🔄 Project Workflow  

### 1. 📂 Data Preparation  
- Load a labeled text dataset (e.g., binary/multiclass)  
- Encode categorical labels into numerical format  
- Split dataset into **training** and **testing** sets  

### 2. 🔠 Tokenization  
- Use Hugging Face's **DistilBERT tokenizer**  
- Convert text into token IDs and attention masks  
- Ensure proper padding and truncation  

### 3. 🧠 Model Implementation - DistilBERT  
- Use `transformers` library to load **`distilbert-base-uncased`**  
- Add a classification head on top (e.g., linear layer)  
- Fine-tune the model using **PyTorch** or **Trainer API**  

### 4. 📊 Evaluation  
- Compute metrics including:  
  - ✅ **Accuracy**  
  - ✅ **Precision & Recall**  
  - ✅ **F1-Score**  
  - ✅ **Confusion Matrix**  
- Visualize results for deeper insights  

### 5. 🔮 Inference  
- Pass custom text inputs to the trained model  
- Return predicted labels with confidence scores  

---

## 🧪 Libraries & Tools  
- 🤗 `transformers` – Hugging Face model & tokenizer  
- 📊 `sklearn` – Evaluation metrics  
- 🧮 `pandas`, `numpy` – Data handling  
- 🔥 `torch` – Model training (or optionally, TensorFlow)

---

## ✍️ Author

**Arpita Mishra**  
B.Tech CSE, [C.V Raman Global University]  
*Passionate about NLP, AI, and deep learning.*


