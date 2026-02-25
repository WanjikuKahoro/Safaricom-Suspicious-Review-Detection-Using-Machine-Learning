# Multilingual Suspicious Review Detection (Kenya)  
**Character-level NLP + behavioural feature engineering for detecting suspicious app-store reviews (English, Swahili, Sheng).**

This project builds and evaluates a machine learning pipeline to flag **potentially deceptive / anomalous reviews** using:
- **Textual signals** (character n-grams for multilingual + code-mixed text)
- **Behavioural signals** (burst activity, helpful-vote patterns, timing)
- **Structural writing patterns** (length, punctuation/emoji density, repetition)
- **Rating–sentiment inconsistencies** (e.g., 1* text that looks strongly positive)

The work is implemented in the notebook: **`Emma.ipynb`**.

---

## Table of contents
- [Problem & context](#problem--context)
- [Data](#data)
- [Approach](#approach)
- [Feature engineering](#feature-engineering)
- [Weak labeling (heuristics)](#weak-labeling-heuristics)
- [Models](#models)
- [Results](#results)
- [How to run](#how-to-run)
- [Repo structure](#repo-structure)
- [Limitations](#limitations)
- [Future work](#future-work)
- [License](#license)

---

## Problem & context
App-store reviews strongly influence product perception and decision-making (ratings, discoverability, roadmap prioritization).  
For Kenyan platforms such as **Safaricom** and **M-Pesa**, reviews are often **multilingual and code-mixed** (English, Swahili, Sheng). This increases the risk of:
- Fake positive reviews
- Coordinated negative attacks
- Spam / repetitive content
- Rating–text contradictions

**Goal:** build a localized, multilingual system that can automatically identify suspicious reviews.

---

## Data
Reviews were scraped from the **Google Play Store** for:
- Safaricom App
- M-Pesa App

Each record includes:
- `review_text`
- `rating` (1–5)
- `timestamp`
- `app_version`
- `helpful_votes` (or similar helpfulness signal)

> The notebook expects the dataset at: `./Data/safaricom_reviews_multilingual.csv`

---

## Approach
End-to-end workflow in the notebook:
1. **Data understanding** (initial inspection)
2. **Cleaning & preprocessing** (missing values, duplicates, timestamp normalization, text cleaning)
3. **EDA** (rating distribution, language distribution, text length distributions, burst activity, term frequency)
4. **Heuristic weak labeling** to create a target (`is_suspicious`)
5. **Progressive modeling**:
   - Baseline: TF-IDF + Logistic Regression
   - Improved: TF-IDF + behavioural/structural features
   - Final: **Character n-grams** + behavioural/structural + sentiment inconsistency
6. **Threshold tuning** for precision-focused operation

---

## Feature engineering
Key engineered features used in later models include:

### Structural features
- Review length (chars/words)
- Uppercase ratio
- Punctuation intensity (e.g., `!!!`)
- Emoji counts (simple regex-based counting)
- Repetition patterns / spam-like structure

### Behavioural features
- Burst activity proxies (multiple reviews in tight time windows)
- Helpful-vote patterns (where available)
- Time-based features derived from the `timestamp`

### Fraud-logic features
- Rating–sentiment mismatch indicators
- Contradiction signals between star rating and lexicon/emoji sentiment

---

## Weak labeling (heuristics)
Because “ground truth” labels for fake/suspicious reviews are rarely available, the notebook creates a weak label:
- Multilingual lexicon-based sentiment scoring (English + Swahili + Sheng + emoji markers)
- Rating–sentiment mismatch rules (e.g., very negative rating but positive sentiment)
- Burst-activity rule(s)
- Combined into a final binary label: `is_suspicious`

This is used as a *proxy* target for supervised learning and comparative evaluation of feature strategies.

---

## Models
All models use **Logistic Regression** for stability and interpretability with sparse/high-dimensional features.

1. **Baseline**: text-only `TfidfVectorizer` (word-level) + Logistic Regression  
2. **Improved**: text TF-IDF + engineered behavioural/structural/fraud features via `ColumnTransformer`  
3. **Final**: **character-level n-grams** + behavioural/structural/fraud features  
4. **Threshold optimization**: precision-focused selection of a decision threshold

---

## Results
Evaluation in the notebook reports classification metrics plus ROC-AUC and PR-AUC.

### Baseline (TF-IDF word-level, threshold=0.50)
- Accuracy: **0.633**
- ROC-AUC: **0.551**
- PR-AUC: **0.180**
- Suspicious class (1): precision **0.141**, recall **0.412**

### Improved (Text + behavioural/structural features, threshold=0.50)
- Accuracy: **0.74**
- ROC-AUC: **0.616**
- PR-AUC: **0.279**
- Suspicious class (1): precision **0.20**, recall **0.42**

### Final (Char n-grams + behavioural/structural/sentiment, threshold=0.50)
- Accuracy: **0.76**
- ROC-AUC: **0.635**
- PR-AUC: **0.357**
- Suspicious class (1): precision **0.23**, recall **0.43**

### Precision-focused threshold tuning (chosen threshold ≈ **0.646**)
- Accuracy: **0.86**
- Suspicious class (1): precision **0.40**, recall **0.29**

**Key takeaway:** adding behavioural/fraud-logic features improves ranking quality (AUCs), and **character n-grams** help handle multilingual + code-mixed text. Threshold tuning allows you to trade recall for precision depending on moderation cost.

---

## Limitations
- **Weak labels** may encode heuristic bias (not true ground truth).
- Review-level behavioural signals are limited without **user/account identifiers**.
- Potential class imbalance and drift across time/app versions.
- Performance metrics reflect the weak-label definition of “suspicious”.

---
