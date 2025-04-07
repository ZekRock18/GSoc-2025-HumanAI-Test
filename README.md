# 🌟 GSoC 2025 — HumanAI Candidate Assessment 🌟
**AI-Powered Behavioral Analysis for Suicide Prevention, Substance Use, and Mental Health Crisis Detection**  
*Longitudinal Geospatial Crisis Trend Analysis* 🚀

---

## 📋 Project Overview

This project is a **GSoC 2025 Candidate Test** designed to showcase skills in:

✅ Extracting, processing, and analyzing crisis-related discussions from social media  
✅ Applying NLP techniques for **sentiment analysis** and **high-risk content detection**  
✅ Geocoding and visualizing crisis trends on an interactive **heatmap**

This is a crucial step toward building AI systems that can **save lives**, identify **mental health crises**, and **direct timely interventions**. 🧠❤️

---

## 🔥 Problem Statement

The challenge is to **extract**, **analyze**, and **visualize** real-time social media discussions related to:

- Mental Health Distress 😟
- Substance Use 🍷🚬
- Suicidality 🆘

Tasks were split into three key phases:

---

## 🛠️ Task 1: Data Extraction & Preprocessing

⏰ **Estimated Time**: 1–1.5 hours

**Goal**: Retrieve social media posts and prepare clean data for NLP!

### Steps:
- Connect to **Twitter/X API** or **Reddit API** 📡
- Extract posts using 10–15 predefined keywords (e.g., `depressed`, `overwhelmed`, `addiction help`, `suicidal`) 🧵
- Save relevant metadata:
  - Post ID
  - Timestamp
  - Content
  - Engagement Metrics (likes, comments, shares) 📊
- Clean the text:
  - Remove stopwords, emojis, and special characters ✨

📦 **Deliverables**:
- Python script to retrieve and store posts
- A cleaned dataset (CSV/JSON) ready for analysis

---

## 💬 Task 2: Sentiment Analysis & Crisis Risk Classification

⏰ **Estimated Time**: 1.5–2 hours

**Goal**: Use NLP models to classify sentiment and detect crisis risk!

### Steps:
- Apply sentiment classification:
  - **VADER** (for Twitter) 🐦 or **TextBlob** 📚
- Detect crisis keywords using:
  - **TF-IDF** or **Word Embeddings** (e.g., BERT, Word2Vec)
- Classify posts into Risk Levels:
  - 🔴 **High-Risk**: Direct crisis language (e.g., "I don't want to be here anymore")
  - 🟠 **Moderate Concern**: Seeking help, discussing struggles (e.g., "I feel lost lately")
  - 🟢 **Low Concern**: General discussions about mental health

📦 **Deliverables**:
- Python script for sentiment and risk classification
- Table or plot showing sentiment/risk distribution 📈

---

## 🗺️ Task 3: Crisis Geolocation & Mapping

⏰ **Estimated Time**: 1–2 hours

**Goal**: Map crisis discussions geospatially for actionable insights!

### Steps:
- Extract location data using:
  - Geotagged posts (if available) 📍
  - NLP-based place recognition (e.g., "Need help in Austin" → maps to Austin, TX) 📌
- Generate a **heatmap** using **Folium** or **Plotly**
- Highlight top 5 locations with highest crisis activity 🔥

📦 **Deliverables**:
- Interactive heatmap displaying crisis trends
- List of top crisis-prone locations

---

## 🧰 Tech Stack

- Python 🐍
- Tweepy / PRAW (Twitter/X and Reddit API Clients)
- NLTK, SpaCy, VADER, TextBlob (NLP & Sentiment Analysis) 🧠
- Scikit-learn, TensorFlow/Keras (optional ML models) 🤖
- Folium, Plotly (Visualization) 🗺️
- Pandas, NumPy (Data Handling) 📊

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/ZekRock18/GSoc-2025-HumanAI-Test.git
   cd GSoc-2025-HumanAI-Test
