import streamlit as st
import joblib
import pandas as pd
import os
import sys

import numpy as np
# We manually re-add the 'int' attribute to numpy so old libraries don't crash
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

# This tells Python to look inside the 'app' folder for feature_engineering.py
current_dir = os.path.dirname(os.path.abspath(__file__)) # This is the 'app' folder
repo_root = os.path.dirname(current_dir) # This is the 'main' folder
sys.path.append(current_dir)

from feature_engineering import build_features

st.set_page_config(page_title="Safaricom Review Audit", layout="centered")

# LOAD ASSETS FROM THE ROOT 'model' FOLDER
@st.cache_resource
def load_assets():
    # We look for the model in the main directory
    model_path = os.path.join(repo_root, "model", "model.joblib")
    
    if not os.path.exists(model_path):
        model_path = "model/model.joblib"
        
    model = joblib.load(model_path)
    threshold = 0.95  # Based on your Pipeline logic
    return model, threshold

try:
    model, threshold = load_assets()
    st.sidebar.success("✅ Model & Logic Loaded")
except Exception as e:
    st.error(f"❌ Path Error: Could not find the model folder. {e}")
    st.stop()

#  UI CODE
st.title("🛡️ Safaricom Review Shield")
st.markdown("Analyzing English, Swahili, and Sheng reviews for anomalies.")

with st.form("input_form"):
    review_text = st.text_area("Review Text", placeholder="Andika hapa...")
    col1, col2 = st.columns(2)
    with col1:
        rating = st.slider("Rating", 1, 5, 3)
        thumbs = st.number_input("Thumbs Up", 0)
    with col2:
        is_mixed = st.toggle("Code Mixed")
        is_sheng = st.toggle("Sheng-like")
    
    submit = st.form_submit_button("Run Detection", use_container_width=True)

if submit:
    # Prepare data
    data = pd.DataFrame([{
        "review_text": review_text,
        "rating": rating,
        "thumbs_up": thumbs,
        "is_code_mixed": int(is_mixed),
        "is_sheng_like": int(is_sheng)
    }])
    
    # Process
    X = build_features(data)
    proba = model.predict_proba(X)[:, 1][0]
    
    # Results
    st.divider()
    if proba >= threshold:
        st.error(f"🚩 **SUSPICIOUS** (Probability: {proba:.2%})")
        st.info(f"Model Threshold is set to {threshold}")
    else:
        st.success(f"✅ **GENUINE** (Probability: {proba:.2%})")