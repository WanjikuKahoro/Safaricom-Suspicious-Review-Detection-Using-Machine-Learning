# Safaricom Suspicious Review Detection System using Machine Learning

## Project Overview

This project develops a machine learning framework for detecting **suspicious reviews** in Google Play Store data for Safaricom and M-Pesa. The objective is to identify potentially manipulative, coordinated, or anomalous reviews using a combination of textual modeling, behavioural signals, anomaly detection, and threshold optimization.

The final system is designed as a **high-precision risk-flagging tool**, suitable for prioritizing suspicious reviews for moderation or investigation rather than making automated fraud decisions.

## Problem Statement

Online review platforms are vulnerable to manipulation. Traditional sentiment analysis alone is insufficient to detect such patterns. Therefore, this project builds a structured suspicious review detection framework combining:
- Text modeling (TF-IDF)
- Behavioural anomaly detection
- Heuristic labeling
- Threshold optimization

## Objectives
### Main Objective
To build a machine learning model that can automatically detect suspicious reviews from Safaricom and M-Pesa Google Play Store, enabling high-confidence flagging of potentially inauthentic reviews for review monitoring and moderation.

#### Specific Objectives
The specific objectives of the project are;
 1. To engineer textual and behavioral features to support suspicious review detection.
 2. To define and label suspicious reviews in the Safaricom and M-Pesa datasets as targets for model training.
 3. To train, evaluate and refine models for accurately detecting high-confidence suspicious reviews.

## Dataset

- Source: Google Play Store reviews  
- Applications: Safaricom and M-Pesa  
- Data includes:
  - Review text
  - Rating
  - Date
  - Engagement (thumbs up count)
---

## Methodology
1. Text Preprocessing
2. Feature Engineering
3. Suspicious Label Construction (Weak Supervision)

## Model Development

Three model configurations were evaluated:

1. Baseline Model
- TF-IDF features only
- Logistic Regression (class-balanced)
- Default threshold = 0.50

2. Hybrid Model
- TF-IDF + behavioural + anomaly features
- Logistic Regression
- Default threshold = 0.50

3. Optimized Threshold Model
- Same as Hybrid Model
- Decision threshold increased to ≈ 0.95
- Designed for high-confidence suspicious detection

### Key Insights
- Behavioural features significantly improved ranking performance (PR-AUC improved from 0.11 to 0.39).
- Default threshold (0.50) produced excessive false positives.
- Raising the threshold to ≈0.95 dramatically improved precision (0.83).
- The optimized model achieves the best suspicious-class F1-score (0.45).

## Limitations

1. Labels are heuristic-based and not verified fraud ground truth.
2. Class imbalance affects minority-class learning.
3. No user-level or device-level behavioural signals.
4. High precision comes at the cost of lower recall.
5. Duplicate detection captures exact repetition but may miss paraphrased campaigns.
6. Sentiment signals are rule-based and may misclassify nuanced complaints.

## Recommendations for Improvement

1. Introduce manual validation to create stronger ground-truth labels.
2. Incorporate user-level behavioural aggregation if data becomes available.
3. Benchmark additional models (SVM, Gradient Boosting, calibrated ensembles).

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib / Seaborn (for visualization)

---