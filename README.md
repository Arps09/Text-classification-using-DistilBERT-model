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

## 📈 Model Performance

| Metric        | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|---------------|---------|---------|-----------|--------------|
| **Precision** | 0.87    | 0.92    | 0.89      | 0.90         |
| **Recall**    | 0.91    | 0.88    | 0.90      | 0.90         |
| **F1-Score**  | 0.89    | 0.90    | 0.90      | 0.90         |
| **Support**   | 653     | 731     | 1384      | 1384         |
| **Accuracy**  |         |         |           | **0.90**     |

---

## 🧪 Libraries & Tools  
- 🤗 `transformers` – Hugging Face model & tokenizer  
- 📊 `sklearn` – Evaluation metrics  
- 🧮 `pandas`, `numpy` – Data handling  
- 🔥 `torch` – Model training (or optionally, TensorFlow)

---

## 🧠 Key Learnings

Throughout this project, I gained valuable insights and hands-on experience in:

- 🔍 **Understanding Transformers**: Learned how the DistilBERT model works under the hood and why it’s an efficient alternative to BERT for text classification tasks.

- 🧼 **Text Preprocessing**: Explored essential preprocessing steps such as tokenization, padding, truncation, and converting labels into numerical format using Hugging Face's tokenizer.

- 🧠 **Model Fine-Tuning**: Successfully fine-tuned `distilbert-base-uncased` on a custom dataset for binary classification using PyTorch and Hugging Face's `Trainer` API.

- 📊 **Evaluation Metrics**: Deepened my understanding of NLP performance metrics like Precision, Recall, F1-Score, and how they help measure the quality of predictions.

- 🚀 **Streamlit Deployment**: Deployed a real-time sentiment analysis tool using Streamlit, allowing users to interact with the model via a web interface.

- 🧪 **End-to-End NLP Pipeline**: Built a complete machine learning pipeline — from loading data to model training, evaluation, and inference — using modern libraries like `transformers`, `sklearn`, and `torch`.

This project helped strengthen my foundation in NLP and gave me practical skills in building and deploying transformer-based models.

---

## 📃License
This project is for educational and research purposes only.

---
## ✍️ Author

**Arpita Mishra**  
B.Tech CSE, [C.V Raman Global University]  
*Passionate about NLP, AI, and deep learning.*


