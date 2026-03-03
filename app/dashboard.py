import streamlit as st
import requests

st.set_page_config(page_title="Safaricom Review Audit", layout="wide")

st.title("🔍 Safaricom Suspicious Review Detector")
st.write("This tool uses your **Logistic Regression (Balanced)** model to flag anomalies.")

# Create two columns: Left for input, Right for results
col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("Input Review Data")
    text = st.text_area("Review Text", "M-Pesa is very good but it keeps crashing! 😡")
    rating = st.slider("Rating", 1, 5, 1)
    thumbs = st.number_input("Thumbs Up", 0, 100, 5)
    
    # Extra features from your schema
    c1, c2 = st.columns(2)
    with c1: mixed = st.checkbox("Code Mixed")
    with c2: sheng = st.checkbox("Sheng-like")
    
    run = st.button("Analyze Pattern", use_container_width=True)

if run:
    payload = {
        "review_text": text,
        "rating": rating,
        "thumbs_up": thumbs,
        "is_code_mixed": mixed,
        "is_sheng_like": sheng
    }
    
    try:
        # Call your local FastAPI (change this URL when you host it online)
        res = requests.post("http://127.0.0.1:8000/predict", json=payload).json()
        
        with col_out:
            st.subheader("Analysis Results")
            prob = res["suspicious_probability"]
            
            # Visual Gauge
            st.metric("Suspicion Probability", f"{prob:.2%}")
            st.progress(prob)
            
            if res["is_suspicious_pred"]:
                st.error("### 🚩 FLAG: SUSPICIOUS")
                st.write(f"Probability exceeds your **{res['threshold']}** threshold.")
            else:
                st.success("### ✅ FLAG: GENUINE")
                st.write("Confidence score is within normal bounds.")
                
            st.info(f"**Insight:** The model sees a high contradiction between the {rating}-star rating and the text sentiment.")
            
    except Exception as e:
        st.error("Ensure your FastAPI server is running at http://127.0.0.1:8000")