# 🌟 SentinelAI: Longitudinal Behavioral and Geospatial Analysis for Mental Health Crisis Detection 🌟
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
  - **TF-IDF**
  - **Word Embeddings** (BERT / Word2Vec)
- Classify posts into Risk Levels:
  - 🔴 **High-Risk**: Direct crisis expressions (e.g., "I don't want to be here anymore")
  - 🟠 **Moderate Concern**: Seeking help or discussing struggles
  - 🟢 **Low Concern**: General discussions about mental health

📦 **Deliverables**:
- Python script that classifies posts based on sentiment and risk
- Table/plot showing distribution by sentiment and risk category

---

## 🗺️ Task 3: Crisis Geolocation & Mapping

⏰ **Estimated Time**: 1–2 hours

**Goal**: Visualize crisis discussions across locations using a heatmap!

### Steps:
- Extract location metadata:
  - From geotagged posts 📍
  - Using NLP place recognition (e.g., "Need help in Austin" → Austin, TX)
- Generate an interactive heatmap with **Folium** or **Plotly**
- Highlight Top 5 Locations 📈

📦 **Deliverables**:
- Heatmap visualization of crisis posts
- List of Top 5 locations with highest crisis mentions

---

## 📄 Proposal Document

You can read the detailed project plan and vision here:

👉 [**GSoC 2025 Proposal — Prakhar Gupta**](https://docs.google.com/document/d/1fak3_rfyA8PrAz3lJ76iI4nob8sCmk85h8YGm1Gtoqg/edit?usp=sharing)  

---

## 🛠️ Tech Stack

- Python 🐍
- Tweepy / PRAW APIs
- VADER / TextBlob for Sentiment Analysis
- TF-IDF / BERT / Word2Vec for NLP
- Folium / Plotly for Visualization

---

## 🤝 Contribution

This project was completed as part of the **GSoC 2025 Selection Process**.  
Grateful for the opportunity to contribute towards meaningful AI solutions in mental health. 🙏

---

# 🚀 Let's Build a Safer World Together!
