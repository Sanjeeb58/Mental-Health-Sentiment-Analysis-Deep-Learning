# Mental Health Sentiment Analysis using Deep Learning (RoBERTa) 🧠💖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Project Overview

In our digital world, prioritizing mental well-being is critical. This project applies advanced **Natural Language Processing (NLP)** techniques to classify mental health-related text into **seven distinct sentiment categories**: *Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal*. 

Using both traditional machine learning (Logistic Regression) and fine-tuned transformer models (**RoBERTa**), this project demonstrates how AI can assist in the early detection of mental health issues from user-generated content.

> **Note:** This is a student research project intended for educational purposes. It is not a substitute for professional medical advice or diagnosis.

## 🧠 Motivation
In the digital era, individuals often express their deepest struggles through online platforms. Sentiment analysis offers a way to detect and interpret these emotional cues, providing:
* 🆘 **Early intervention** for at-risk individuals.
* 🌐 **Public mental health insights** to shape policies.
* ❤️ **Stigma reduction** through empathetic AI.
* 🧠 **Tailored support systems** via classification-driven responses.

## 🔍 Dataset
* **Source:** Kaggle - Sentiment Analysis for Mental Health
* **Size:** Over 50,000 labeled mental health text entries.
* **Classes:** 7 multi-class categories (Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal).
* **Data Split:** * Training Set: 42,434 samples
    * Test Set: 10,609 samples

## 🔧 Methodology

### 1. Data Preprocessing
A robust preprocessing pipeline cleaned and normalized text data:
* **Text Cleaning:** Expanded contractions, removed special tokens (URLs, hashtags), handled digits, and removed punctuation.
* **Linguistic Processing:** Stopword removal and lemmatization (using spaCy/NLTK).
* **Filtering:** Removed excessively short texts (<5 words).

### 2. Exploratory Data Analysis (EDA)
Performed comprehensive analysis including Sentiment Distribution Charts (Bar/Pie) and N-gram Plots (Unigrams, Bigrams, Trigrams) to understand textual patterns.

### 3. Models Implemented
* **Logistic Regression (Baseline):** Uses TF-IDF vectorization with `f1_weighted` scoring.
* **RoBERTa (Fine-Tuned Transformer):** * Base model: `roberta-base`
    * Optimizer: AdamW with linear warmup.
    * Loss Function: CrossEntropyLoss with balanced class weights.
    * Performance: Trained for 7 epochs with early stopping.

## 📈 Results

| Model | Accuracy | F1 Score |
| :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 71.00% | 0.71 |
| **RoBERTa (Fine-Tuned)** | **75.33%** | **0.75** |

## ✨ Prediction Examples

The model processes raw text and outputs the predicted sentiment along with probability distributions.

**Example 1: Normal Sentiment**
> *Input:* "I am feeling absolutely ecstatic and overjoyed today, everything is wonderful!"
>
> *Prediction:* **Normal** (Probability: 75.8%)

**Example 2: Depression Sentiment**
> *Input:* "The weight of this sadness is crushing me. I feel so empty and depressed."
>
> *Prediction:* **Depression** (Probability: 97.4%)

## 🚀 Installation & Running

To replicate this project locally:

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Sanjeeb58/Mental-Health-Sentiment-Analysis-Deep-Learning.git](https://github.com/Sanjeeb58/Mental-Health-Sentiment-Analysis-Deep-Learning.git)
   cd Mental-Health-Sentiment-Analysis-Deep-Learning

   Install required libraries
   
2. **Install required texts**
   ```bash
   pip install -r requirements.txt
  (Note: Ensure you have PyTorch installed with CUDA support if using GPU)

3. **Run the Notebook**
   ```bash
    jupyter notebook "Mental_Health_Sentiment_Analysis_(RoBERTa).ipynb"

🤝 Future Work
This project is open for further improvements. Future goals include:
Expanding the Dataset: Incorporating more diverse sources to reduce bias.
Model Exploration: Experimenting with BERT-Large or XLNet.
Deployment: Creating a web API (using Flask/FastAPI) for real-time inference.

📄 License
This project is open-source and available under the MIT License.
