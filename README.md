# Disaster Tweet Classification

## Overview
This repository contains a **Natural Language Processing (NLP) project** that classifies tweets as disaster-related or not. The goal is to build a robust machine learning model that can automatically detect disaster-related tweets, which can be useful for emergency response teams and social media monitoring.

## Dataset
The dataset used in this project is from Kaggle's **"Real or Not? NLP with Disaster Tweets"** competition. It consists of labeled tweets:
- `1` (Disaster-related tweet)
- `0` (Non-disaster tweet)

The dataset is preprocessed using **NLTK (Natural Language Toolkit)** for cleaning and tokenization.

## Features & Methods Used
### **Text Preprocessing**
- Removing **URLs, mentions, hashtags, numbers, and special characters**
- Converting text to **lowercase**
- Tokenization and **lemmatization** (using NLTK)
- Removing **stopwords**

### **Feature Engineering**
- **TF-IDF Vectorization** (with **unigrams, bigrams, and trigrams**)
- **Word2Vec Embeddings** (for deep contextual understanding)
- **BERT Embeddings** (using DistilBERT for enhanced accuracy)

### **Model Training & Evaluation**
Several models are trained and evaluated using **F1-Score**, **Precision**, and **Recall**:
1. **Logistic Regression** (with balanced class weights)
2. **Naive Bayes Classifier**
3. **Random Forest**
4. **Ensemble Model (Logistic Regression + Naïve Bayes)**
5. **Deep Learning Models (Word2Vec + BERT Embeddings)**

## Results
| Model | F1-Score |
|--------|----------|
| Logistic Regression | **0.935** |
| Naive Bayes | **0.943** |

## Submission
The final predictions are stored in `submission.csv`, formatted as required by Kaggle.

## Future Improvements
- Fine-tuning **BERT** embeddings for better context understanding.
- Implementing **LSTMs/Transformers** for sequence modeling.
- Exploring **hyperparameter tuning** for improved performance.

## Author
**Rama Charan** – AI/ML Enthusiast | NLP Researcher
